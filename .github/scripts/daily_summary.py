import os
import subprocess
import requests
import json
from datetime import datetime, timedelta

def run_command(cmd):
    # Use encoding utf-8 explicitly to avoid garbled logs on diverse runners
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()

def get_recent_commits(branch):
    if branch and branch.lower() != 'all':
        print(f"Fetching commits for branch: {branch}")
        cmd = f"git log origin/{branch} --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    else:
        print("Fetching commits for ALL branches...")
        cmd = "git log --all --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    
    latest_commits = run_command(cmd)
    return latest_commits

def call_openclaw_llm(commits_data, branch):
    api_key = os.environ.get("OPENCLAW_API_KEY", "")
    if not api_key:
        print("Warning: OPENCLAW_API_KEY is empty. Outputting raw git logs only.")
        return None

    url = "https://ai.gs88.shop/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    branch_text = f"[{branch}] branch" if branch and branch.lower() != 'all' else "[All branches]"
    prompt = f"""
请作为一位资深的研发技术主管，对{branch_text}过去 24 小时内的代码提交记录和修改内容进行深度总结汇报。

【强制格式和内容要求】：
1. 核心业务概括（一段话，不超过 50 字）：用最简洁精准的非技术性大白话，一句话总结这段时间主要解决了什么问题，或者上线了什么核心变化（例如：“重点修复了首页白屏崩溃问题，并优化了底层查询算法”）。
2. 具体修改点列举（分点说明）：将琐碎的代码提交记录按照逻辑聚合成功能模块，并简洁清晰地列出每一个改动。不要像机械翻译一样罗列每个 commit 记录，要融合成自然逻辑。
3. 工作量点评（一两句话评述）：对整体变更的代码活跃度或工作量进行点评。

以下是过去 24 小时内的原始代码变更数据（注意去粗取精）：
{commits_data[:10000]}
"""

    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "你是一个高效、精练、深刻洞察技术变更逻辑的代码审计专家。请返回清晰、干净的纯文本或者 markdown，并且使用规范的中文字符。"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        # Attempt to explicitly instruct response decode format to utf8
        response.encoding = 'utf-8'
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error analyzing with AI, details: {str(e)}")
        return None

def main():
    target_branch = os.environ.get("TARGET_BRANCH", "all").strip() or "all"
    print(f"Gathering worklogs from the last 24 hours for branch: {target_branch}...")
    
    commits = get_recent_commits(target_branch)
    
    report_dir = "branch_summaries"
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    branch_name_for_file = "all_branches" if target_branch.lower() == 'all' else target_branch.replace("/", "_")
    report_file = os.path.join(report_dir, f"summary_{branch_name_for_file}_{date_str}.md")
    
    branch_text = f"【{target_branch}】分支" if target_branch.lower() != 'all' else "【所有分支】"

    if not commits:
        content = f"## {branch_text} 在这 24 小时内没有任何代码提交。\n\n大家辛苦了！好好休息！"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("No commits found.")
        return
        
    print(f"Found following commits:\n{commits[:500]}...\n")
    
    summary = call_openclaw_llm(commits, target_branch)
    
    if summary:
        content = f"# {branch_text} 变更总结 ({date_str})\n\n{summary}"
    else:
        content = f"# {branch_text} 原始修改记录 ({date_str})\n\n```text\n{commits}\n```\n\n*(配置有效正确的 OPENCLAW_API_KEY 及 AI url 后获取智能化概括)*"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Report generated at:", report_file)

if __name__ == "__main__":
    main()
