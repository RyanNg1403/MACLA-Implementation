import re
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ALFWorldLikeLoader:
    """Loads and parses ALFWorld-like text logs"""

    trajectory_pattern = re.compile(r"ID=(\w+_\d+)")
    task_pattern = re.compile(r"Task:\s*(.*)")
    action_pattern = re.compile(r"Action:\s*(.*)")
    observation_pattern = re.compile(r"Observation:\s*(.*)")
    success_pattern = re.compile(r"Success:\s*(True|False)")
    reward_pattern = re.compile(r"Reward:\s*([0-9\.\-]+)")

    def load_files(self, file_paths: List[str], include_trajectory_paths: bool = True) -> List[Dict]:
        all_traj: List[Dict] = []
        for fp in file_paths:
            logger.info(f"Loading from: {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    content = f.read()
                trajs = self._parse_alfworld_content(content, fp)
                if include_trajectory_paths:
                    for t in trajs:
                        t["source_file"] = fp
                        t["file_type"] = Path(fp).stem
                all_traj.extend(trajs)
                logger.info(f"Loaded {len(trajs)} trajectories")
            except FileNotFoundError:
                logger.error(f"File not found: {fp}")
            except Exception as e:
                logger.error(f"Error loading {fp}: {e}")
        logger.info(f"Total trajectories: {len(all_traj)}")
        return all_traj

    def _parse_alfworld_content(self, content: str, source_file: str = "") -> List[Dict]:
        trajs: List[Dict] = []
        blocks = re.split(r"(?=ID=\w+_\d+)", content.strip())
        for block in blocks:
            b = block.strip()
            if not b:
                continue
            t = self._parse_single_trajectory(b, source_file)
            if t:
                trajs.append(t)
        return trajs

    def _parse_single_trajectory(self, block: str, source_file: str = "") -> Optional[Dict]:
        lines = block.split("\n")
        traj = {
            "id": "",
            "task": "",
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": False,
            "think_steps": [],
            "raw_text": block,
            "source_file": source_file,
            "trajectory_path": [],
        }
        try:
            idm = self.trajectory_pattern.search(lines[0])
            if idm:
                traj["id"] = idm.group(1)

            step_index = 0
            current_obs = ""
            for line in lines:
                line = line.strip()
                tm = self.task_pattern.search(line)
                if tm:
                    traj["task"] = tm.group(1)

                if line.startswith("Think:"):
                    traj["think_steps"].append(line[6:].strip())

                am = self.action_pattern.search(line)
                if am:
                    action = am.group(1)
                    traj["actions"].append(action)
                    traj["trajectory_path"].append(
                        {
                            "step": step_index,
                            "action": action,
                            "observation": current_obs,
                            "think": traj["think_steps"][-1] if traj["think_steps"] else "",
                        }
                    )
                    step_index += 1

                om = self.observation_pattern.search(line)
                if om:
                    current_obs = om.group(1)
                    traj["observations"].append(current_obs)

                sm = self.success_pattern.search(line)
                if sm:
                    traj["success"] = sm.group(1) == "True"

                rm = self.reward_pattern.search(line)
                if rm:
                    try:
                        traj["reward"] = float(rm.group(1))
                    except Exception:
                        pass

            for i in range(min(len(traj["actions"]), len(traj["observations"]))):
                traj["action_observation_pairs"].append(
                    {"action": traj["actions"][i], "observation": traj["observations"][i]}
                )

            if "reward" not in traj:
                traj["reward"] = 1.0 if traj["success"] else 0.0

            return traj
        except Exception as e:
            logger.error(f"Parse error in {source_file}: {e}")
            return None


class WebShopLoader:
    """Loads and parses WebShop data - e-commerce domain"""

    def load_and_split_webshop(
        self,
        file_path: str,
        train_ratio: float = 0.7,
        val_seen_ratio: float = 0.15,
        val_unseen_ratio: float = 0.15
    ) -> Dict[str, List[Dict]]:
        """Load WebShop data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading WebShop data from: {file_path}")

        all_trajectories = self._parse_webshop_file(file_path)

        if not all_trajectories:
            logger.error("No WebShop trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}

        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)

        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]

        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }

        logger.info(f"WebShop split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")

        return splits

    def _parse_webshop_file(self, file_path: str) -> List[Dict]:
        """Parse WebShop conversational JSON format"""
        trajectories = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                episodes = data
            elif isinstance(data, dict) and "episodes" in data:
                episodes = data["episodes"]
            else:
                logger.error("Unknown WebShop format")
                return []

            logger.info(f"Found {len(episodes)} WebShop episodes")

            for episode in episodes:
                traj = self._parse_webshop_conversation(episode)
                if traj:
                    trajectories.append(traj)

        except Exception as e:
            logger.error(f"Error parsing WebShop file: {e}")

        logger.info(f"Successfully parsed {len(trajectories)} WebShop trajectories")
        return trajectories

    def _parse_webshop_conversation(self, episode: Dict) -> Optional[Dict]:
        """Parse single WebShop conversational episode"""
        episode_id = episode.get("id", "unknown")
        conversations = episode.get("conversations", [])
        reward = episode.get("reward", 0.0)

        if not conversations:
            return None

        traj = {
            "id": f"webshop_{episode_id}",
            "task": "",
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": reward >= 1.0,
            "think_steps": [],
            "raw_text": json.dumps(episode),
            "source_file": "webshop",
            "trajectory_path": [],
            "domain": "webshop",
            "reward": float(reward)
        }

        for conv in conversations:
            if conv.get("from") == "human":
                value = conv.get("value", "")
                if "Instruction:" in value and "[SEP]" in value:
                    parts = value.split("[SEP]")
                    for i, part in enumerate(parts):
                        if "instruction:" in part.lower():
                            if i + 1 < len(parts):
                                traj["task"] = parts[i + 1].strip()
                                break
                    break

        if not traj["task"]:
            for conv in conversations:
                if conv.get("from") == "gpt":
                    action_match = re.search(r"Action:\s*search\[(.*?)\]", conv.get("value", ""))
                    if action_match:
                        traj["task"] = f"find {action_match.group(1)}"
                        break

        step_index = 0
        current_thought = ""

        for i, conv in enumerate(conversations):
            role = conv.get("from", "")
            value = conv.get("value", "")

            if role == "gpt":
                thought_match = re.search(r"Thought:\s*(.*?)(?:\n|Action:|$)", value, re.DOTALL)
                action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", value)

                if thought_match:
                    current_thought = thought_match.group(1).strip()
                    traj["think_steps"].append(current_thought)

                if action_match:
                    action = action_match.group(1).strip()
                    traj["actions"].append(action)

                    observation = ""
                    if i + 1 < len(conversations) and conversations[i + 1].get("from") == "human":
                        obs_value = conversations[i + 1].get("value", "")
                        if "Observation:" in obs_value:
                            obs_match = re.search(r"Observation:\s*(.*)", obs_value, re.DOTALL)
                            if obs_match:
                                observation = obs_match.group(1).strip()
                                if "[SEP]" in observation:
                                    observation = observation.split("[SEP]")[0].strip()

                    traj["observations"].append(observation)

                    traj["trajectory_path"].append({
                        "step": step_index,
                        "action": action,
                        "observation": observation,
                        "think": current_thought
                    })

                    traj["action_observation_pairs"].append({
                        "action": action,
                        "observation": observation
                    })

                    step_index += 1
                    current_thought = ""

        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory for episode {episode_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None

        return traj


class TravelPlannerLoader:
    """Loads and parses TravelPlanner data - trip planning domain"""

    def load_and_split_travelplanner(
        self,
        file_path: str,
        train_ratio: float = 0.7,
        val_seen_ratio: float = 0.15,
        val_unseen_ratio: float = 0.15
    ) -> Dict[str, List[Dict]]:
        """Load TravelPlanner data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading TravelPlanner data from: {file_path}")

        all_trajectories = self._parse_travelplanner_file(file_path)

        if not all_trajectories:
            logger.error("No TravelPlanner trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}

        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)

        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]

        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }

        logger.info(f"TravelPlanner split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")

        return splits

    def _parse_travelplanner_file(self, file_path: str) -> List[Dict]:
        """Parse TravelPlanner JSON-lines format"""
        trajectories = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        traj = self._parse_travelplanner_trajectory(data)
                        if traj:
                            trajectories.append(traj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error at line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error parsing TravelPlanner file: {e}")

        logger.info(f"Successfully parsed {len(trajectories)} TravelPlanner trajectories")
        return trajectories

    def _parse_travelplanner_trajectory(self, data: Dict) -> Optional[Dict]:
        """Parse single TravelPlanner trajectory"""
        traj_id = data.get("id", "unknown")
        trajectory_text = data.get("trajectory_text", "")
        success = data.get("Success", False)

        if not trajectory_text:
            return None

        task = self._extract_task(trajectory_text)
        if not task:
            logger.warning(f"No task found in trajectory {traj_id}")
            return None

        traj = {
            "id": f"travelplanner_{traj_id}",
            "task": task,
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": success,
            "think_steps": [],
            "raw_text": trajectory_text,
            "source_file": "travelplanner",
            "trajectory_path": [],
            "domain": "travelplanner",
            "reward": 1.0 if success else 0.0,
            "split": data.get("split", "unknown"),
            "synthetic": data.get("synthetic", False)
        }

        lines = trajectory_text.split('\n')
        step_index = 0
        current_think = ""
        current_action = ""

        for line in lines:
            line = line.strip()

            if line.startswith("Think:"):
                current_think = line[6:].strip()
                traj["think_steps"].append(current_think)

            elif line.startswith("Action:"):
                action_text = line[7:].strip()
                action_match = re.match(r'(\w+)\s*(\{.*\})?', action_text)
                if action_match:
                    action_name = action_match.group(1)
                    action_params = action_match.group(2) or ""
                    current_action = f"{action_name}{action_params}"
                    traj["actions"].append(current_action)
                else:
                    current_action = action_text
                    traj["actions"].append(current_action)

            elif line.startswith("Observation:"):
                observation = line[12:].strip()
                traj["observations"].append(observation)

                if current_action:
                    traj["trajectory_path"].append({
                        "step": step_index,
                        "action": current_action,
                        "observation": observation,
                        "think": current_think
                    })

                    traj["action_observation_pairs"].append({
                        "action": current_action,
                        "observation": observation
                    })

                    step_index += 1
                    current_think = ""
                    current_action = ""

        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory {traj_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None

        return traj

    def _extract_task(self, trajectory_text: str) -> str:
        """Extract task description from trajectory text"""
        task_match = re.search(
            r'Your task is to:\s*(.*?)(?:\n|Think:)',
            trajectory_text,
            re.DOTALL
        )

        if task_match:
            task = task_match.group(1).strip()
            task = re.sub(r'\s+', ' ', task)
            return task

        lines = trajectory_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['plan', 'trip', 'travel', 'from', 'to']):
                return line.strip()

        return ""


class SQLLoader:
    """Loads and parses InterCode SQL data - database query domain"""

    def load_and_split_sql(
        self,
        file_path: str,
        train_ratio: float = 0.7,
        val_seen_ratio: float = 0.15,
        val_unseen_ratio: float = 0.15,
        use_gold_as_success: bool = True
    ) -> Dict[str, List[Dict]]:
        """Load SQL data and split into train/val_seen/val_unseen"""
        logger.info(f"Loading SQL data from: {file_path}")

        all_trajectories = self._parse_sql_file(file_path, use_gold_as_success)

        if not all_trajectories:
            logger.error("No SQL trajectories loaded")
            return {"train": [], "val_seen": [], "val_unseen": []}

        success_count = sum(1 for t in all_trajectories if t.get('success', False))
        logger.info(f"SQL Success Rate: {success_count}/{len(all_trajectories)} ({success_count/len(all_trajectories)*100:.1f}%)")

        random.seed(42)
        indices = list(range(len(all_trajectories)))
        random.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_val_seen = int(len(indices) * val_seen_ratio)

        train_idx = indices[:n_train]
        val_seen_idx = indices[n_train:n_train + n_val_seen]
        val_unseen_idx = indices[n_train + n_val_seen:]

        splits = {
            "train": [all_trajectories[i] for i in train_idx],
            "val_seen": [all_trajectories[i] for i in val_seen_idx],
            "val_unseen": [all_trajectories[i] for i in val_unseen_idx]
        }

        logger.info(f"SQL split - Train: {len(splits['train'])}, "
                   f"Val Seen: {len(splits['val_seen'])}, Val Unseen: {len(splits['val_unseen'])}")

        return splits

    def _parse_sql_file(self, file_path: str, use_gold_as_success: bool = True) -> List[Dict]:
        """Parse SQL JSON-lines format"""
        trajectories = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        traj = self._parse_sql_trajectory(data, use_gold_as_success)
                        if traj:
                            trajectories.append(traj)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parse error at line {line_num}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error parsing SQL file: {e}")

        logger.info(f"Successfully parsed {len(trajectories)} SQL trajectories")
        return trajectories

    def _parse_sql_trajectory(self, data: Dict, use_gold_as_success: bool = True) -> Optional[Dict]:
        """Parse single SQL trajectory with gold query as learning target"""
        traj_id = data.get("id", "unknown")
        instruction = data.get("instruction", "")
        db = data.get("db", "unknown")
        gold_query = data.get("gold", "")
        actual_success = data.get("success", False)
        steps = data.get("steps", [])

        if not instruction:
            logger.warning(f"No instruction in SQL trajectory {traj_id}")
            return None

        if use_gold_as_success and gold_query:
            success = True
            reward = 1.0
        else:
            success = actual_success
            reward = 1.0 if actual_success else 0.0

        schema_info = steps[0].get("observation", "") if steps else ""
        tables = self._extract_tables(schema_info)
        sql_operations = self._extract_sql_operations(gold_query)

        traj = {
            "id": f"sql_{traj_id}",
            "task": instruction,
            "actions": [],
            "observations": [],
            "action_observation_pairs": [],
            "success": success,
            "think_steps": [],
            "raw_text": json.dumps(data),
            "source_file": "intercode_sql",
            "trajectory_path": [],
            "domain": "sql",
            "reward": reward,
            "db": db,
            "hardness": data.get("hardness", "unknown"),
            "gold_query": gold_query,
            "actual_success": actual_success,
            "sql_tables": tables,
            "sql_operations": sql_operations
        }

        if use_gold_as_success and gold_query:
            step = {
                "step": 0,
                "action": gold_query,
                "observation": schema_info,
                "observation_next": "Query executed successfully (synthetic)",
                "think": f"Generate SQL query to answer: {instruction}",
                "done": True
            }

            traj["actions"].append(gold_query)
            traj["observations"].append(schema_info)
            traj["think_steps"].append(step["think"])
            traj["trajectory_path"].append(step)
            traj["action_observation_pairs"].append({
                "action": gold_query,
                "observation": schema_info,
                "observation_next": step["observation_next"]
            })
        else:
            for step_index, step in enumerate(steps):
                observation = step.get("observation", "")
                think = step.get("think", "")
                action = step.get("action", "")
                observation_next = step.get("observation_next", "")

                if think:
                    traj["think_steps"].append(think)

                if action:
                    traj["actions"].append(action)

                if observation:
                    traj["observations"].append(observation)

                traj["trajectory_path"].append({
                    "step": step_index,
                    "action": action,
                    "observation": observation,
                    "observation_next": observation_next,
                    "think": think,
                    "done": step.get("done", False)
                })

                traj["action_observation_pairs"].append({
                    "action": action,
                    "observation": observation,
                    "observation_next": observation_next
                })

        if not traj["task"] or not traj["actions"]:
            logger.warning(f"Incomplete trajectory {traj_id}: task='{traj['task']}', actions={len(traj['actions'])}")
            return None

        return traj

    def _extract_tables(self, schema_text: str) -> List[str]:
        """Extract table names from schema observation"""
        tables = []
        for match in re.finditer(r'(\w+)\([^)]+\)', schema_text):
            table_name = match.group(1)
            if table_name not in ['Tables', 'sqlite_sequence']:
                tables.append(table_name)
        return list(set(tables))

    def _extract_sql_operations(self, sql_query: str) -> List[str]:
        """Extract SQL operations and patterns"""
        if not sql_query:
            return []

        operations = []
        sql_upper = sql_query.upper()

        if 'SELECT' in sql_upper:
            operations.append('select')
        if 'FROM' in sql_upper:
            operations.append('from')
        if 'WHERE' in sql_upper:
            operations.append('where')
        if 'JOIN' in sql_upper or 'INNER JOIN' in sql_upper or 'LEFT JOIN' in sql_upper:
            operations.append('join')
        if 'GROUP BY' in sql_upper:
            operations.append('group_by')
        if 'ORDER BY' in sql_upper:
            operations.append('order_by')
        if 'HAVING' in sql_upper:
            operations.append('having')
        if 'LIMIT' in sql_upper:
            operations.append('limit')
        if 'COUNT' in sql_upper:
            operations.append('count')
        if 'SUM' in sql_upper:
            operations.append('sum')
        if 'AVG' in sql_upper:
            operations.append('avg')
        if 'MAX' in sql_upper:
            operations.append('max')
        if 'MIN' in sql_upper:
            operations.append('min')
        if 'DISTINCT' in sql_upper:
            operations.append('distinct')
        if 'UNION' in sql_upper:
            operations.append('union')
        if 'SUBQUERY' in sql_upper or '(SELECT' in sql_upper:
            operations.append('subquery')

        return operations
