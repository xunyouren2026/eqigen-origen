# utils/script_generator.py
"""
全自动脚本生成器：从故事创意生成专业镜头脚本（JSON格式）
支持本地 LLM（HuggingFace）或 OpenAI API，内置镜头语法规则，输出可直接用于视频生成的 JSON。
"""

import json
import os
from typing import Dict, List, Optional, Any

# 可选依赖导入
try:
    import jsonschema
except ImportError:
    jsonschema = None
    print("警告: jsonschema 未安装，将跳过 JSON 格式校验，建议安装: pip install jsonschema")

# 本地 LLM 导入（可选）
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安装，将无法使用本地模型，请安装: pip install transformers")

# ============================================================================
# 镜头语法规则库（可扩展）
# ============================================================================
LENS_RULES = {
    "紧张": {
        "shot_type": "close_up",
        "camera_motion": "dolly",
        "rhythm": "fast",
        "duration": 1.5,
        "lighting": "dramatic",
        "color_tone": "high_contrast"
    },
    "浪漫": {
        "shot_type": "medium",
        "camera_motion": "tilt",
        "rhythm": "slow",
        "duration": 3.0,
        "lighting": "soft",
        "color_tone": "warm"
    },
    "宏大": {
        "shot_type": "extreme_long",
        "camera_motion": "crane",
        "rhythm": "slow",
        "duration": 4.0,
        "lighting": "natural",
        "color_tone": "neutral"
    },
    "悲伤": {
        "shot_type": "close_up",
        "camera_motion": "static",
        "rhythm": "slow",
        "duration": 3.0,
        "lighting": "soft",
        "color_tone": "cool"
    },
    "欢乐": {
        "shot_type": "medium",
        "camera_motion": "pan",
        "rhythm": "fast",
        "duration": 2.0,
        "lighting": "natural",
        "color_tone": "warm"
    },
    "惊悚": {
        "shot_type": "extreme_close",
        "camera_motion": "handheld",
        "rhythm": "fast",
        "duration": 1.2,
        "lighting": "dramatic",
        "color_tone": "high_contrast"
    },
    "neutral": {
        "shot_type": "medium",
        "camera_motion": "static",
        "rhythm": "normal",
        "duration": 2.5,
        "lighting": "natural",
        "color_tone": "neutral"
    }
}

# 默认镜头字段（用于补全缺失值）
DEFAULT_SHOT = {
    "duration": 2.5,
    "shot_type": "medium",
    "camera_motion": "static",
    "focus": [0.5, 0.5],
    "lighting": "natural",
    "color_tone": "neutral",
    "rhythm": "normal",
    "transition": "cut"
}

# JSON Schema（用于校验，可选）
SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "duration": {"type": "number", "minimum": 0.5, "maximum": 30.0},
                    "shot_type": {"type": "string", "enum": ["extreme_long", "long", "medium", "close_up", "extreme_close"]},
                    "camera_motion": {"type": "string", "enum": ["static", "pan", "tilt", "zoom", "dolly", "track", "crane", "handheld"]},
                    "focus": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "lighting": {"type": "string", "enum": ["natural", "dramatic", "soft", "hard"]},
                    "color_tone": {"type": "string", "enum": ["neutral", "warm", "cool", "sepia", "high_contrast"]},
                    "rhythm": {"type": "string", "enum": ["slow", "normal", "fast"]},
                    "transition": {"type": "string", "enum": ["cut", "fade", "dissolve", "wipe", "zoom"]}
                },
                "required": ["duration", "shot_type", "camera_motion"]
            }
        }
    },
    "required": ["shots"]
}

# ============================================================================
# 故事分析器（使用 LLM 提取关键情节）
# ============================================================================
class StoryAnalyzer:
    """使用 LLM 分析故事，提取情节点及其情绪、时长等"""
    def __init__(self, generator):
        self.generator = generator

    def analyze(self, story: str, max_points: int = 8) -> List[Dict[str, Any]]:
        """
        调用 LLM 分析故事，返回情节点列表
        每个情节点包含：description, emotion, duration（秒）
        """
        prompt = f"""你是一个专业的电影编剧。请分析以下故事，提取关键情节点，并为每个情节点分配情绪标签和预估时长（秒）。
情绪标签从以下选择：紧张、浪漫、宏大、悲伤、欢乐、惊悚、neutral。
输出格式为 JSON 列表，每个元素包含：description（简短描述）、emotion、duration（数值）。
只输出 JSON，不要其他文字。

故事：{story}

最多输出 {max_points} 个情节点。"""
        response = self.generator._call_llm(prompt)
        try:
            # 尝试解析 JSON
            points = json.loads(response)
            if isinstance(points, dict) and "points" in points:
                points = points["points"]  # 兼容可能包装的对象
            # 确保是列表
            if not isinstance(points, list):
                points = [points]
            # 限制数量
            points = points[:max_points]
            # 补全字段
            for p in points:
                p.setdefault("emotion", "neutral")
                p.setdefault("duration", 2.5)
            return points
        except json.JSONDecodeError:
            # 降级：返回一个默认情节点
            return [{"description": story[:50], "emotion": "neutral", "duration": 10.0}]

# ============================================================================
# 镜头规划器（基于规则）
# ============================================================================
class LensPlanner:
    """根据情节点生成镜头序列"""
    def __init__(self, rules: Dict = None):
        self.rules = rules or LENS_RULES

    def plan(self, plot_points: List[Dict]) -> List[Dict]:
        shots = []
        for idx, point in enumerate(plot_points):
            emotion = point.get("emotion", "neutral")
            rule = self.rules.get(emotion, self.rules["neutral"])
            # 生成一个镜头
            shot = {
                "duration": point.get("duration", rule["duration"]),
                "shot_type": rule["shot_type"],
                "camera_motion": rule["camera_motion"],
                "focus": [0.5, 0.5],  # 默认中心
                "lighting": rule.get("lighting", "natural"),
                "color_tone": rule.get("color_tone", "neutral"),
                "rhythm": rule.get("rhythm", "normal"),
                "transition": "cut" if idx == 0 else "cut"  # 默认硬切
            }
            # 根据情节变化调整过渡（例如高潮处用溶解）
            if emotion in ["紧张", "惊悚"] and idx > 0:
                shot["transition"] = "fade"
            shots.append(shot)
        return shots

# ============================================================================
# 后处理与校验
# ============================================================================
def validate_and_fix_script(script: Dict) -> Dict:
    """确保脚本符合规范，补全缺失字段"""
    # 确保有 shots 字段
    if "shots" not in script:
        script["shots"] = []
    # 对每个镜头补全默认字段
    for shot in script["shots"]:
        for key, default in DEFAULT_SHOT.items():
            if key not in shot:
                shot[key] = default
        # 确保 focus 是长度为2的列表
        if "focus" not in shot:
            shot["focus"] = [0.5, 0.5]
        elif len(shot["focus"]) != 2:
            shot["focus"] = [0.5, 0.5]
    # 可选：使用 jsonschema 校验
    if jsonschema:
        try:
            jsonschema.validate(script, SCHEMA)
        except jsonschema.ValidationError as e:
            print(f"脚本校验警告: {e.message}")
    return script

# ============================================================================
# 主类：ScriptGenerator
# ============================================================================
class ScriptGenerator:
    """
    脚本生成器主类
    支持本地 HuggingFace 模型或 OpenAI API
    """
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 use_local: bool = True,
                 api_key: Optional[str] = None,
                 max_plot_points: int = 8,
                 custom_rules: Optional[Dict] = None):
        """
        :param model_name: 模型名称（本地 HuggingFace 模型 ID 或 OpenAI 模型名如 gpt-4）
        :param use_local: 是否使用本地模型（需安装 transformers）
        :param api_key: OpenAI API 密钥（仅当 use_local=False 时使用）
        :param max_plot_points: 最多生成的情节点数
        :param custom_rules: 自定义镜头规则，覆盖默认规则
        """
        self.model_name = model_name
        self.use_local = use_local
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_plot_points = max_plot_points
        self.rules = {**LENS_RULES, **(custom_rules or {})}

        if use_local and not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers 未安装，无法使用本地模型。请安装: pip install transformers")

        if not use_local and not self.api_key:
            raise ValueError("使用 API 模式需要提供 api_key 或设置 OPENAI_API_KEY 环境变量")

        # 初始化 LLM 管道（本地模式）
        self.pipeline = None
        if use_local:
            try:
                self.pipeline = pipeline("text-generation", model=model_name, device_map="auto")
            except Exception as e:
                print(f"加载本地模型失败: {e}")
                self.pipeline = None
                raise

        # 初始化子模块
        self.analyzer = StoryAnalyzer(self)
        self.planner = LensPlanner(self.rules)

    def generate(self, story: str, style: str = "cinematic") -> Dict:
        """
        生成镜头脚本
        :param story: 故事文本
        :param style: 风格（目前保留，可扩展）
        :return: 符合格式的 JSON 脚本
        """
        # 1. 分析故事，提取情节点
        plot_points = self.analyzer.analyze(story, max_points=self.max_plot_points)
        # 2. 规划镜头
        shots = self.planner.plan(plot_points)
        # 3. 封装脚本
        script = {"shots": shots}
        # 4. 校验与修复
        script = validate_and_fix_script(script)
        return script

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 并返回原始文本"""
        if self.use_local:
            if not self.pipeline:
                raise RuntimeError("本地模型未正确初始化")
            # 生成响应
            outputs = self.pipeline(prompt, max_new_tokens=1024, do_sample=False, temperature=0.2)
            generated_text = outputs[0]["generated_text"]
            # 尝试提取 prompt 之后的内容
            if prompt in generated_text:
                return generated_text.split(prompt)[-1].strip()
            return generated_text.strip()
        else:
            # 调用 OpenAI 风格 API
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 1024
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"API 调用失败: {resp.text}")
            return resp.json()["choices"][0]["message"]["content"]

    def generate_to_file(self, story: str, output_path: str, style: str = "cinematic") -> str:
        """
        生成脚本并保存为 JSON 文件
        :return: 保存的文件路径
        """
        script = self.generate(story, style)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(script, f, indent=2, ensure_ascii=False)
        return output_path

# ============================================================================
# 简单命令行测试（可选）
# ============================================================================
if __name__ == "__main__":
    # 示例：使用本地模型生成脚本
    # 需要设置环境变量 OPENAI_API_KEY 或修改 use_local=True
    generator = ScriptGenerator(use_local=False, api_key="your-api-key-here")  # 替换为实际密钥
    story = "一个年轻的探险家在丛林中迷失，他必须找到回家的路。途中他遇到了一只友善的豹子，他们一起克服困难，最终成功返回村庄。"
    script = generator.generate(story)
    print(json.dumps(script, indent=2, ensure_ascii=False))