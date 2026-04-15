import os
import subprocess
import requests
import json
from datetime import datetime, timedelta

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_recent_commits(branch):
    # 如果指定了分支，则获取该分支最近 24 小时的 commits，反之获取全部分支的
    if branch:
        print(f"Fetching commits for branch: {branch}")
        cmd = f"git log origin/{branch} --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    else:
        cmd = "git log --all --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    
    latest_commits = run_command(cmd)
    return latest_commits

def call_openclaw_llm(commits_data, branch):
    # 模拟 OpenClaw / AI 模型 Base 调用的实际 URL（如要用 OpenClaw 或者大模型，替换为真实可用的 Endpoint）
    api_key = os.environ.get("OPENCLAW_API_KEY", "")
    if not api_key:
        print("警告: 环境变量 OPENCLAW_API_KEY 为空。系统将只会输出原始 Git 日志，不进行 AI 总结。")
        return None

    # 此处 URL 为示范，可以是开源通用的 openai 兼容格式 API。如果用通义千问/文心一言/DeepSeek 请换地址和模型名
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
请作为一位资深的研发技术主管，对【{branch}】分支过去 24 小时内的代码提交记录和修改内容进行深度总结汇报。

【强制格式和内容要求】：
1. 核心业务概括（一段话，不超过 50 字）：用最简洁精准的非技术性大白话，一句话总结这个分支在这段时间主要解决了什么问题，或者上线了什么核心变化（例如：“重点修复了首页白屏崩溃问题，并优化了底层查询算法”）。
2. 具体修改点列举（分点说明）：将琐碎的代码提交记录按照逻辑聚合成功能模块，并简洁清晰地列出每一个改动。不要像机械翻译一样罗列每个 commit 记录，要融合成自然逻辑。
3. 工作量点评（一两句话评述）：对整体变更的代码活跃度或工作量进行点评。

以下是过去 24 小时分支的原始代码变更数据（注意去粗取精）：
{commits_data[:10000]}
"""

    payload = {
        "model": "gpt-4o-mini", # 请确保您在使用的 AI 服务上该模型名存在
        "messages": [
            {"role": "system", "content": "你是一个高效、精练、深刻洞察技术变更逻辑的代码审计专家。"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"调用 AI 分析出错 details: {str(e)}")
        return None

def main():
    # 从 Workflow 的输入中获取目标分支名称，默认为 master
    target_branch = os.environ.get("TARGET_BRANCH", "master")
    print(f"Gathering worklogs from the last 24 hours for branch: {target_branch}...")
    
    commits = get_recent_commits(target_branch)
    
    report_dir = "branch_summaries"
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.join(report_dir, f"summary_{target_branch}_{date_str}.md")
    
    if not commits:
        content = f"## {target_branch} 分支在这 24 小时内没有任何代码提交。\n\n大家辛苦了！好好休息！"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("无提交记录。")
        return
        
    print(f"找到以下提交:\n{commits[:500]}...\n")
    
    # 调用 OpenClaw / 大模型 进行智能凝练和总结
    summary = call_openclaw_llm(commits, target_branch)
    
    if summary:
        content = f"# 【{target_branch}】分支变更总结 ({date_str})\n\n{summary}"
    else:
        # 降级：如果没有 AI 服务，直接呈现记录
        content = f"# 【{target_branch}】分支原始修改记录 ({date_str})\n\n```text\n{commits}\n```\n\n*(配置有效正确的 OPENCLAW_API_KEY 及 AI url 后获取智能化概括)*"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Report generated at:", report_file)

if __name__ == "__main__":
    main()
