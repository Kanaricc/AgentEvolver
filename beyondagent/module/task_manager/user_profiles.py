from dataclasses import dataclass
from typing import List


@dataclass
class EnvEntityOpt:
    """定义实体可执行的操作"""
    name: str
    description: str


def get_crud_opts() -> List[EnvEntityOpt]:
    """返回通用的 CRUD 操作集"""
    return [
        EnvEntityOpt("create", "创建该实体的新实例"),
        EnvEntityOpt("read", "查询或检索该实体的一个或多个属性值"),
        EnvEntityOpt("update", "修改该实体的一个或多个属性值"),
        EnvEntityOpt("delete", "删除该实体的实例")
    ]


@dataclass
class EnvEntity:
    """环境中的信息实体"""
    name: str
    description: str
    attrs: dict[str, str]  # 属性名 -> 描述
    opts: List[EnvEntityOpt]


class TaskPreference:
    """表示用户希望生成的任务特征"""
    def __init__(self, num_entities: int, num_opts: int, relation_difficulty: float):
        self._num_entities = num_entities
        self._num_opts = num_opts
        self._relation_difficulty = relation_difficulty
        assert self._relation_difficulty>=1 and self._relation_difficulty<=3

    @property
    def num_entities(self) -> int:
        return self._num_entities

    @property
    def num_opts(self) -> int:
        return self._num_opts

    @property
    def relation_difficulty(self) -> str:
        """将难度数值映射为文字描述"""
        mapping = {
            1: (
                "简单：仅涉及一个实体或一个属性，"
                "无需跨实体或跨属性的关联操作，"
                "可以通过单步操作完成。"
            ),
            2: (
                "中等：涉及多个实体或属性，"
                "但这些操作之间相互独立，"
                "可以并行理解，无需前后依赖条件判断。"
            ),
            3: (
                "困难：涉及多个实体或属性，"
                "且操作必须先结合属性判断条件，"
                "或后续操作依赖前一步的结果，"
                "需要推理和决策。"
            )
        }
        assert self._relation_difficulty>=1 and self._relation_difficulty<=3
        return mapping[int(self._relation_difficulty)]


class UserProfile:
    """用户档案及任务生成器"""
    def __init__(self, name: str, background: str, task: TaskPreference):
        self._name = name
        self._background = background
        self._entities: List[EnvEntity] = []
        self._task_preference = task

    def reg_entity(self, entity: EnvEntity):
        self._entities.append(entity)

    def reg_entities(self, entities: List[EnvEntity]):
        self._entities.extend(entities)

    def get_instruction(self) -> str:
        """
        生成一份详细的 LLM 指令，精确描述各个部分，
        让模型能更好地生成符合要求的 query
        """
        inst_parts = []
        
        inst_parts.append("# 角色及环境信息")
        # 角色设定
        inst_parts.append("### 角色设定")
        inst_parts.append(
            f"你是一个智能任务生成助手，名字是 {self._name}，你的背景信息：{self._background}。"
            "你能够理解环境中的信息实体、属性和可执行的操作，并基于这些信息在环境中自由探索：尝试使用 API来操作相应实体。"
        )

        # 实体信息
        inst_parts.append("\n### 环境实体信息")
        for e in self._entities:
            inst_parts.append(f"- 实体：{e.name} — {e.description}")
            for attr_name, attr_desc in e.attrs.items():
                inst_parts.append(f"  - 属性：{attr_name} — {attr_desc}")
            inst_parts.append("  - 可执行操作：")
            for opt in e.opts:
                inst_parts.append(f"    - {opt.name}：{opt.description}")

        # 任务偏好
        inst_parts.append("\n### 偏好")
        inst_parts.append(f"- 涉及的平均实体数量：{self._task_preference.num_entities}")
        inst_parts.append(f"- 涉及的平均操作数量：{self._task_preference.num_opts}")
        inst_parts.append(f"- 关系难度：{self._task_preference.relation_difficulty}")

        # 开始任务
        inst_parts.append("\n### 开始你的工作")
        inst_parts.append(
            "现在，请充分应用上述信息，并开始探索环境吧。"
        )

        return "\n".join(inst_parts)


# ===== 示例使用 =====
if __name__ == "__main__":
    song_entity = EnvEntity(
        name="歌曲",
        description="音乐收藏中的歌曲条目",
        attrs={"标题": "歌曲的名称", "星级": "用户对歌曲的评分"},
        opts=get_crud_opts() + [EnvEntityOpt("play", "播放该歌曲")]
    )

    account_entity = EnvEntity(
        name="账号",
        description="用户的个人账户",
        attrs={"名字": "账户名称", "余额": "账户的当前余额"},
        opts=get_crud_opts()
    )

    task_pref = TaskPreference(num_entities=2, num_opts=2, relation_difficulty=3)

    user = UserProfile(
        name="小明",
        background="音乐爱好者，喜欢根据心情播放歌曲",
        task=task_pref
    )

    user.reg_entities([song_entity, account_entity])

    print(user.get_instruction())