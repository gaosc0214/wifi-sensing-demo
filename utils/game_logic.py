import time
from typing import List, Dict, Any

class GameSession:
    """游戏会话管理类"""
    
    def __init__(self):
        self.total_score = 0
        self.total_games = 0
        self.correct_guesses = 0
        self.game_history = []
        self.start_time = time.time()
    
    @property
    def accuracy(self) -> float:
        """计算准确率"""
        if self.total_games == 0:
            return 0.0
        return self.correct_guesses / self.total_games
    
    def add_result(self, user_correct: bool, score: int):
        """添加游戏结果"""
        self.total_games += 1
        self.total_score += score
        
        if user_correct:
            self.correct_guesses += 1
        
        # 记录游戏历史
        self.game_history.append({
            'game_number': self.total_games,
            'user_correct': user_correct,
            'score': score,
            'timestamp': time.time()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取游戏统计信息"""
        current_time = time.time()
        session_duration = current_time - self.start_time
        
        return {
            'total_score': self.total_score,
            'total_games': self.total_games,
            'accuracy': self.accuracy,
            'session_duration': session_duration,
            'average_score_per_game': self.total_score / max(1, self.total_games),
            'recent_performance': self.get_recent_performance()
        }
    
    def get_recent_performance(self, last_n_games: int = 5) -> Dict[str, float]:
        """获取最近N局的表现"""
        if not self.game_history:
            return {'accuracy': 0.0, 'average_score': 0.0}
        
        recent_games = self.game_history[-last_n_games:]
        
        correct_count = sum(1 for game in recent_games if game['user_correct'])
        total_score = sum(game['score'] for game in recent_games)
        
        return {
            'accuracy': correct_count / len(recent_games),
            'average_score': total_score / len(recent_games)
        }
    
    def reset(self):
        """重置游戏会话"""
        self.total_score = 0
        self.total_games = 0
        self.correct_guesses = 0
        self.game_history = []
        self.start_time = time.time()

def calculate_score(user_correct: bool, ai_correct: bool) -> int:
    """
    计算游戏得分
    
    Args:
        user_correct: 用户是否猜对
        ai_correct: AI是否预测正确
        
    Returns:
        score: 得分
    """
    if user_correct and ai_correct:
        # 双方都对：基础分
        return 20
    elif user_correct and not ai_correct:
        # 用户对，AI错：高分奖励
        return 30
    elif not user_correct and ai_correct:
        # 用户错，AI对：安慰分
        return 5
    else:
        # 双方都错：小额安慰分
        return 10

def get_achievement_level(total_score: int) -> Dict[str, Any]:
    """
    根据总分获取成就等级
    
    Args:
        total_score: 总分
        
    Returns:
        achievement info: 成就信息
    """
    if total_score >= 1000:
        return {
            'level': 'CSI大师',
            'emoji': '🏆',
            'description': '您已成为CSI感知领域的专家！',
            'color': 'gold'
        }
    elif total_score >= 500:
        return {
            'level': 'WiFi专家',
            'emoji': '🥇',
            'description': '对WiFi感知有很深的理解！',
            'color': 'silver'
        }
    elif total_score >= 200:
        return {
            'level': '信号猎手',
            'emoji': '🥈',
            'description': '能够识别大部分活动模式！',
            'color': 'bronze'
        }
    elif total_score >= 50:
        return {
            'level': '数据探索者',
            'emoji': '🥉',
            'description': '开始理解CSI数据的奥秘！',
            'color': 'blue'
        }
    else:
        return {
            'level': '新手',
            'emoji': '🔰',
            'description': '继续探索WiFi的隐形世界！',
            'color': 'green'
        }

def get_performance_feedback(accuracy: float, recent_accuracy: float) -> str:
    """
    根据表现给出反馈
    
    Args:
        accuracy: 总体准确率
        recent_accuracy: 最近的准确率
        
    Returns:
        feedback: 反馈信息
    """
    if recent_accuracy > accuracy + 0.1:
        return "🔥 状态火热！您正在快速进步！"
    elif recent_accuracy > 0.8:
        return "💪 表现优秀！您对CSI模式的理解很深刻！"
    elif recent_accuracy > 0.6:
        return "👍 不错的表现！继续保持！"
    elif recent_accuracy > 0.4:
        return "📚 还有进步空间，多观察CSI模式的细节！"
    else:
        return "💡 别灰心！每次尝试都是学习的机会！"

def generate_challenge_hint(activity_id: int, confidence: float) -> str:
    """
    根据活动和AI置信度生成提示
    
    Args:
        activity_id: 活动ID
        confidence: AI预测置信度
        
    Returns:
        hint: 提示信息
    """
    activity_hints = {
        0: "跳跃活动通常在CSI数据中产生突发性的强烈变化",
        1: "跑步活动具有周期性特征，观察重复的模式",
        2: "静坐呼吸是微动检测，信号变化很微妙",
        3: "走路活动有规律的步态特征，注意周期性",
        4: "挥手是局部运动，影响特定的子载波"
    }
    
    base_hint = activity_hints.get(activity_id, "观察CSI数据的整体模式和变化")
    
    if confidence > 0.9:
        confidence_hint = "AI对这个预测非常有信心！"
    elif confidence > 0.7:
        confidence_hint = "AI比较确信这个预测。"
    elif confidence > 0.5:
        confidence_hint = "AI有些不确定，这可能是个有挑战性的样本。"
    else:
        confidence_hint = "AI也很困惑，这个样本很有挑战性！"
    
    return f"💡 {base_hint}\n🤖 {confidence_hint}"

class LeaderBoard:
    """排行榜管理类"""
    
    def __init__(self):
        self.scores = []
    
    def add_score(self, player_name: str, score: int, accuracy: float):
        """添加分数记录"""
        self.scores.append({
            'player': player_name,
            'score': score,
            'accuracy': accuracy,
            'timestamp': time.time()
        })
        # 保持排行榜大小
        self.scores = sorted(self.scores, key=lambda x: x['score'], reverse=True)[:10]
    
    def get_top_scores(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取前N名分数"""
        return self.scores[:n]
    
    def get_rank(self, score: int) -> int:
        """获取分数排名"""
        rank = 1
        for record in self.scores:
            if score >= record['score']:
                return rank
            rank += 1
        return rank

def create_sample_challenge():
    """创建示例挑战数据"""
    import numpy as np
    
    # 生成不同活动的模拟CSI数据
    challenges = []
    
    activity_patterns = {
        0: "jumping",    # 跳跃 - 突发变化
        1: "running",    # 跑步 - 周期性
        2: "breathing",  # 呼吸 - 微变化
        3: "walking",    # 走路 - 规律步态
        4: "waving"      # 挥手 - 局部变化
    }
    
    for activity_id, pattern in activity_patterns.items():
        # 根据活动类型生成特征
        if pattern == "jumping":
            # 跳跃：随机突发
            data = np.random.normal(0, 1, (500, 232))
            jump_times = np.random.choice(500, 5, replace=False)
            for t in jump_times:
                data[max(0, t-5):min(500, t+5), :] += np.random.normal(0, 3, (min(10, 500-max(0, t-5)), 232))
        
        elif pattern == "running":
            # 跑步：周期性信号
            t = np.linspace(0, 10, 500)
            base_signal = np.sin(2 * np.pi * 2 * t)  # 2Hz 步频
            data = np.tile(base_signal.reshape(-1, 1), (1, 232))
            data += np.random.normal(0, 0.5, (500, 232))
        
        elif pattern == "breathing":
            # 呼吸：低频微变化
            t = np.linspace(0, 10, 500)
            base_signal = 0.2 * np.sin(2 * np.pi * 0.3 * t)  # 0.3Hz 呼吸频率
            data = np.tile(base_signal.reshape(-1, 1), (1, 232))
            data += np.random.normal(0, 0.1, (500, 232))
        
        elif pattern == "walking":
            # 走路：规律步态
            t = np.linspace(0, 10, 500)
            base_signal = np.sin(2 * np.pi * 1.5 * t)  # 1.5Hz 步频
            data = np.tile(base_signal.reshape(-1, 1), (1, 232))
            data += np.random.normal(0, 0.3, (500, 232))
        
        else:  # waving
            # 挥手：部分子载波有变化
            data = np.random.normal(0, 0.2, (500, 232))
            # 影响特定子载波范围
            affected_carriers = slice(50, 100)
            t = np.linspace(0, 10, 500)
            wave_signal = np.sin(2 * np.pi * 3 * t)  # 3Hz 挥手
            data[:, affected_carriers] += np.tile(wave_signal.reshape(-1, 1), (1, 50))
        
        challenges.append({
            'activity_id': activity_id,
            'data': data,
            'pattern': pattern
        })
    
    return challenges 