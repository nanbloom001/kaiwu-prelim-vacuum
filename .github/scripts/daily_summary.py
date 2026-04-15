import os
import subprocess
import requests
import json
from datetime import datetime, timedelta

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_recent_commits():
    # 获取过去24小时的所有分支commit
    latest_commits = run_command("git log --all --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status")
    return latest_commits

def call_openclaw_llm(commits_data):
    # 模拟 OpenClaw 或大模型的基础调用 (假设是 OpenAI 兼容接口，请根据实际情况调整 URL 和参数)
    api_key = os.environ.get("OPENCLAW_API_KEY", "")
    if not api_key:
        print("未找到 API KEY，只生成纯文本总结（不调用AI）。")
        return None

    # 此处 URL 为示范，如果是具体的 openclaw 工具的 API 地址请替换
    url = "https://api.openai.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
请作为一位资深的研发工程师，总结以下过去 24 小时在各个分支的代码提交记录。
格式要求：
1. 按照分支/功能模块分类总结。
2. 总结出做了哪些主要的新特性/Bug修复。
3. 评估这段时间的整体工作量和代码变更活跃度。

以下是过去 24 小时的代码变更数据：
{commits_data[:10000]} # 防止太长超出 token 限制
"""

    payload = {
        "model": "gpt-4o-mini", # 根据 openclaw 的模型替换
        "messages": [
            {"role": "system", "content": "你是一个代码审计与Git日志总结专家。"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("调用 AI 分析出错:", str(e))
        return None

def main():
    print("Gathering branch worklogs from the last 24 hours...")
    commits = get_recent_commits()
    
    report_dir = "branch_summaries"
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_file = os.path.join(report_dir, f"summary_{date_str}.md")
    
    if not commits:
        content = "## 过去 24 小时没有任何代码提交。大家辛苦了！好好休息！"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("无提交记录。")
        return
        
    print(f"找到以下提交:\n{commits[:500]}...\n")
    
    # 尝试调用 OpenClaw（或兼容 LLM）总结
    summary = call_openclaw_llm(commits)
    
    if summary:
        content = f"## 过去 24 小时分支协同进度报告\n\n{summary}"
    else:
        # 降级：如果未配置 API KEY，直接列出原始 commits
        content = f"## 过去 24 小时原始代码提交记录\n\n```text\n{commits}\n```\n\n*(配置正确的 OPENCLAW_API_KEY 后以获取 AI 总结报告)*"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Report generated at:", report_file)

if __name__ == "__main__":
    main()
