# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 14:39:58 2025

@author: USER
"""

import pandas as pd
import numpy as np
import random
from events import EventManager

def load_titanic_data(file_path='train_and_test2.csv'):
    """
    載入簡化版 Titanic 資料，並處理欄位對應與基本清洗。
    對應欄位名稱為：['2urvived', 'Pclass', 'Passengerid', 'Sex', 'Age', 'sibsp', 'Parch', 'Fare', 'Embarked']
    """
    df = pd.read_csv(file_path)

    # 顯示欄位名稱（方便除錯）
    print("目前欄位：", df.columns.tolist())

    # 對應欄位名稱（讓它符合遊戲程式預期）
    df.rename(columns={
        '2urvived': 'Survived',
        'sibsp': 'SibSp',
        'Passengerid': 'Name'  # 用乘客編號假裝角色名稱
    }, inplace=True)

    # 補齊缺失年齡
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # 保留必要欄位
    df = df[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df.dropna(subset=['Survived', 'Pclass', 'Name', 'Sex'], inplace=True)

    return df

def generate_role(df, selected_class):
    """
    從指定艙等中，隨機挑一位乘客作為玩家角色。
    """
    filtered_df = df[df['Pclass'] == selected_class]
    passenger = filtered_df.sample(1).iloc[0]
    
    role_info = {
        'Name': passenger['Name'],
        'Survived': int(passenger['Survived']),
        'Pclass': int(passenger['Pclass']),
        'Sex': passenger['Sex'],
        'Age': passenger['Age'],
        'SibSp': passenger['SibSp'],
        'Parch': passenger['Parch'],
        'Fare': passenger['Fare'],
        'Embarked': passenger['Embarked']
    }
    
    return role_info


def calculate_base_survival_prob(role):
    """
    根據該角色在真實歷史中的Survived欄位（0或1），
    給予一個「原始生還率」作為基礎。
    
    這裡提供兩種簡易示範做法，請依需求自行調整：
    1) 如果在歷史資料中存活(Survived=1) → 給予 70% 基礎存活率
       如果在歷史資料中死亡(Survived=0) → 給予 30% 基礎存活率
    2) 亦可依照艙等、性別等因素綜合計算更擬真。
    """
    if role['Survived'] == 1:
        return 0.7
    else:
        return 0.3

def get_class_name(pclass):
    """將艙等數字轉成對應敘述文字"""
    if pclass == 1:
        return "頭等艙"
    elif pclass == 2:
        return "二等艙"
    else:
        return "三等艙"

def introduce_character(role):
    """
    遊戲開場介紹角色資訊。
    """
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print("【你的角色】")
    print(f"姓名：{role['Name']}")
    print(f"性別：{('男' if role['Sex'] == 'male' else '女')}")
    print(f"年齡：{int(role['Age'])} 歲")
    print(f"艙等：{get_class_name(role['Pclass'])}")
    print(f"票價：{role['Fare']:.2f}")
    print(f"登船港口：{role['Embarked'] if pd.notna(role['Embarked']) else '未知'}")
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝\n")
    input("按下 Enter 繼續...\n")

def generate_event(round_number):
    """
    每回合生成一個「事件」，並提供若干「選項」。
    這裡以簡單範例示意，可以擴充劇情與邏輯。
    """
    # 範例事件池：可依想像力擴增
    event_pool = [
        {
            "title": "船體撞上冰山，海水正在湧入！你決定：",
            "choices": {
                "A": "留在原地等待指示",
                "B": "主動尋找救生艇位置",
                "C": "嘗試逃往頭等艙區域，看看能否找到生路"
            },
            "fate_effects": {
                "A": 0,     # 等待可能沒有立即性改變
                "B": 0.03,  # 提前找到救生艇，有一點正面效果
                "C": 0.02   # 有點幫助，但可能引發更多風險
            },
            "moral_effects": {
                "A": 0,
                "B": 1,   # 積極行動
                "C": -1   # 可能冒犯其他艙等旅客
            }
        },
        {
            "title": "你發現有人試圖賄賂船員爭取上艇的名額。你要：",
            "choices": {
                "A": "湊錢加入賄賂行列",
                "B": "報警抓人，制止不公平行為",
                "C": "袖手旁觀，不想惹麻煩"
            },
            "fate_effects": {
                "A": 0.05,  # 負面行為，但有助於上艇
                "B": 0,     # 英勇舉動，但未必對自己有利
                "C": 0
            },
            "moral_effects": {
                "A": -2,
                "B": 2,
                "C": 0
            }
        },
        {
            "title": "甲板上出現騷亂，士兵在維持秩序，你會：",
            "choices": {
                "A": "混在群眾中試圖強行擠上救生艇",
                "B": "服從命令，排隊等候",
                "C": "利用混亂掩護，假冒高階乘客身份"
            },
            "fate_effects": {
                "A": 0.03,
                "B": -0.02, # 太乖有可能錯失時機
                "C": 0.05   # 假冒成功有機會提高存活，但也有道德風險
            },
            "moral_effects": {
                "A": -1,
                "B": 1,
                "C": -2
            }
        },
        {
            "title": "你聽到隔壁艙室有人呼救，似乎被困在倒塌的傢俱下。你：",
            "choices": {
                "A": "衝去幫忙救援",
                "B": "告訴附近船員，自己先去找出口",
                "C": "怕浪費時間，假裝沒聽見"
            },
            "fate_effects": {
                "A": -0.02, # 浪費黃金時間，存活機率下降
                "B": 0,
                "C": 0.01
            },
            "moral_effects": {
                "A": 2,
                "B": 0,
                "C": -2
            }
        }
    ]
    # 從事件池中隨機選一個事件
    event = random.choice(event_pool)
    return event

def play_round(round_number, role, fate_mod, moral_score, event_manager):
    event = event_manager.get_event_for_round(round_number)
    if event is None:
        print("⚠️ 沒有可用事件了，本回合略過。")
        return fate_mod, moral_score, {}

    print(f"\n第 {round_number} 回合事件：\n{event['title']}")

    for key, desc in event['choices'].items():
        print(f"{key}. {desc}")

    choice = input("請選擇（A/B/C）：").upper()
    while choice not in ["A", "B", "C"]:
        choice = input("請選擇（A/B/C）：").upper()

    # 修改開始：根據選擇套用影響
    fate_change = event['fate_effects'][choice]
    moral_change = event['moral_effects'][choice]
    tag = event['tags'][choice] if 'tags' in event else "未知"

    fate_mod += fate_change
    moral_score += moral_change

    print(f"→ 你展現了「{tag}」的一面。")
    event_manager.set_followup(event, choice)
    input("\n按下 Enter 繼續...\n")

    decision_record = {
        'round': round_number,
        'title': event['title'],
        'choice': choice,
        'text': event['choices'][choice],
        'tag': tag
    }

    return fate_mod, moral_score, decision_record


    
    # 顯示回饋
    print(f"你選擇了 {choice}：{event['choices'][choice]}")
    if fate_change > 0:
        print(f"→ 你的生還機率稍微提升 (+{fate_change*100:.0f}%)！")
    elif fate_change < 0:
        print(f"→ 你的生還機率略為降低 ({fate_change*100:.0f}%)...")
    else:
        print("→ 似乎沒有立即影響，但結果尚難預料。")
    if moral_change > 0:
        print(f"→ 道德立場提高 (+{moral_change})")
    elif moral_change < 0:
        print(f"→ 道德立場降低 ({moral_change})")
    else:
        print("→ 道德立場沒有明顯變化。")
    
    input("\n按下 Enter 繼續...\n")
    
    return fate_mod, moral_score

def determine_ending(final_survival_prob, moral_score):
    """
    根據最終的生還機率與道德分數，決定敘事性多結局。
    
    可自行增修判斷條件及敘述文字。
    """
    # 先判斷是否在機率上存活
    survive_threshold = 0.5  # 假設 > 0.5 即視為「成功上艇」；真實可用random做更隨機的判斷
    survived = (final_survival_prob > survive_threshold)
    
    # 再根據 moral_score 判斷道德取向
    if moral_score >= 5:
        moral_label = "英勇"
    elif moral_score >= 0:
        moral_label = "普通"
    else:
        moral_label = "黑暗"
    
    # 綜合判斷
    if survived:
        if moral_label == "英勇":
            return "【結局】你最終登上救生艇，並竭盡所能協助他人。你雖然活下來了，也贏得人們的尊敬。"
        elif moral_label == "普通":
            return "【結局】你成功活著離開鐵達尼號。雖然過程有掙扎，但至少你保持了基本的良知。"
        else:
            return "【結局】你靠著不擇手段奪下位置，最終存活。不過，你將永遠背負著良心不安。"
    else:
        if moral_label == "英勇":
            return "【結局】你雖然葬身海底，但你的英勇與無私精神，留存在倖存者與後世的記憶裡。"
        elif moral_label == "普通":
            return "【結局】命運無法改寫，你與鐵達尼號一同沉沒。或許這正是歷史注定的結局。"
        else:
            return "【結局】你的各種陰謀算計仍無法逃過最終劫數。沉沒的海底成了你黑暗選擇的終點。"
   


def select_difficulty():
    print("請選擇遊戲難度（對應初始艙等）：")
    print("1. 頭等艙（Easy）")
    print("2. 二等艙（Normal）")
    print("3. 三等艙（Hard）")
    
    level = input("輸入數字 1～3：")
    while level not in ["1", "2", "3"]:
        level = input("請重新輸入（1/2/3）：")
    return int(level)

from collections import Counter

def generate_novel_ending(survived, tag_list):
    tag_counter = Counter(tag_list)
    top_tags = tag_counter.most_common(3)
    tag_phrase = "、".join([f"「{tag}」" for tag, _ in top_tags])

    tone = "你在驚濤駭浪中倖存，這段旅程沒有殺死你，但它改變了你。" if survived else "你最終沉沒於海底，但你留下的痕跡深植人心。"

    personality = ""
    if "自私" in tag_counter and tag_counter["自私"] >= 2:
        personality = "你做了不少爭議性的選擇，在關鍵時刻選擇保全自己，甚至犧牲他人。"
    elif "勇敢" in tag_counter:
        personality = "你展現了難能可貴的勇氣，即便在混亂中仍願意幫助他人。"
    elif "冷靜" in tag_counter:
        personality = "你保持冷靜並做出務實選擇，這使你在混亂中更容易掌握機會。"
    elif "溫柔" in tag_counter:
        personality = "你總是選擇與人同在，付出關懷，讓人安心。"
    else:
        personality = "你的選擇讓人難以評價，也許你只是隨波逐流地活下來。"

    return f"""\n＝＝＝＝＝＝＝📖 你的災難旅程總結 📖＝＝＝＝＝＝＝
{tone}
這十回合中，你的性格特質主要是：{tag_phrase}
{personality}
這段旅程雖已結束，但你所展現的面貌，將永遠烙印在歷史中。
"""

def main():
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print("　🎮《TITANIC：命運沉沒之時》")
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝\n")

    input("按下 Enter 開始遊戲...\n")

    # 1. 載入資料
    df = load_titanic_data('train_and_test2.csv')

    # 2. 玩家選擇艙等（難度）
    selected_class = select_difficulty()

    # 3. 從指定艙等中抽出角色
    role = generate_role(df, selected_class)

    # 4. 介紹角色
    introduce_character(role)
    event_manager = EventManager("events.csv")

    # 5. 設定初始命運狀態
    base_survival_prob = calculate_base_survival_prob(role)
    fate_mod = 0.0
    moral_score = 0

    print(f"系統提示：你在歷史紀錄中的「原始生還率」約為 {base_survival_prob*100:.1f}%")
    print("但命運能否改寫，就看你這十回合的抉擇了...\n")
    input("按下 Enter 進入回合選擇...\n")

    decision_log = []

    # 6. 進行 10 回合
    for round_num in range(1, 11):
        fate_mod, moral_score, decision = play_round(round_num, role, fate_mod, moral_score, event_manager)
        decision_log.append(decision)

    # 7. 計算最終生還率
    final_survival_prob = max(0, min(1, base_survival_prob + fate_mod))

    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print("　　最終結果")
    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝")
    print(f"你的最終生還機率：{final_survival_prob*100:.1f}%")
    print(f"你的最終道德分數：{moral_score}")

    # 8. 顯示標準結局
    ending_text = determine_ending(final_survival_prob, moral_score)
    print(ending_text)

    # 9. 選項回顧
    print("\n＝＝＝＝＝＝＝你這十回合的所有選擇＝＝＝＝＝＝＝")
    for i, log in enumerate(decision_log, 1):
        if isinstance(log, dict) and all(k in log for k in ['title', 'choice', 'text', 'tag']):
            print(f"第 {i} 回合｜{log['title']} 👉 你選擇了：{log['choice']}. {log['text']}（特質：{log['tag']}）")
        else:
            print(f"第 {i} 回合｜⚠️ 無效記錄，可能是事件不足而略過。")

    # 10. 小說式人格總結
    tag_list = [log['tag'] for log in decision_log if 'tag' in log]
    novel_ending = generate_novel_ending(final_survival_prob > 0.5, tag_list)
    print(novel_ending)

    print("＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝\n")
    input("感謝遊玩！按下 Enter 結束遊戲。\n")

if __name__ == "__main__":
    main()


