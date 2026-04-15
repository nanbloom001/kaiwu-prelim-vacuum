import os
import subprocess
import requests
from datetime import datetime

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8')
    return result.stdout.strip()

def get_recent_commits(branch):
    if branch and branch.lower() != 'all':
        cmd = f"git log origin/{branch} --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    else:
        cmd = "git log --all --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    return run_command(cmd)

def get_updated_branches():
    cmd = "git branch -r | grep -v '\\->'"
    branches = []
    output = run_command(cmd)
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('origin/'):
            branch_name = line.replace('origin/', '', 1)
            if branch_name.lower() != 'head':
                branches.append(branch_name)
    return branches

def call_openclaw_llm(prompt):
    api_key = os.environ.get("OPENCLAW_API_KEY", "")
    if not api_key:
        print("Warning: OPENCLAW_API_KEY is empty.")
        return None

    url = "https://ai.gs88.shop/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-5.4",
        "messages": [
            {"role": "system", "content": "你是一个严谨且逻辑清晰的研发代码审计专家，只输出客观技术事实。"},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        # Increase timeout from 60 to 180 to accomodate larger summarizations
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.encoding = 'utf-8'
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"AI API Error: {str(e)}")
        # Print part of the body if available to help identify token limits API rejections.
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

def main():
    target_branch = os.environ.get("TARGET_BRANCH", "all").strip() or "all"
    date_str = datetime.now().strftime("%Y-%m-%d")
    report_dir = os.path.join("branch_summaries", date_str)
    os.makedirs(report_dir, exist_ok=True)
    print(f"Directory created: {report_dir}")

    branches_to_check = [target_branch] if target_branch.lower() != 'all' else get_updated_branches()
    
    has_any_commits = False
    all_branch_details = []

    for branch in branches_to_check:
        commits = get_recent_commits(branch)
        if not commits:
            continue
            
        print(f"Step 1: Extracting changes for branch: {branch}...")
        has_any_commits = True
        
        prompt_branch = f"""
请将以下【{branch}】分支在过去 24 小时内的代码提交记录，整理为一个清晰的 Markdown 列表文档。
【要求】：只罗列由于代码变更所对应的具体修改点（模块/功能/文件级变动），可以适度聚类，不需要写前言后语，不需要整体总结。

======
原始记录:
{commits[:6000]}
"""
        branch_md = call_openclaw_llm(prompt_branch)
        if not branch_md:
            branch_md = f"*(AI调用失败，原始记录参考)*\n```text\n{commits}\n```"
            
        safe_branch_name = branch.replace("/", "_")
        branch_file = os.path.join(report_dir, f"{safe_branch_name}.md")
        with open(branch_file, "w", encoding="utf-8") as f:
            f.write(f"### 分支 {branch} 修改事项：\n\n{branch_md}")
            
        all_branch_details.append(f"--- 【分支：{branch}】 ---\n{branch_md}")

    if not has_any_commits:
        print("未检测到 24 小时内的任何代码变更。")
        no_file = os.path.join(report_dir, "no_updates.md")
        with open(no_file, "w", encoding="utf-8") as f:
            f.write(f"在 {date_str} 过去的 24 小时内，仓库中没有代码提交记录。大家辛苦了！")
        return

    print("Step 2: Generating the overall aggregated summary from individual branches...")
    combined_docs = "\n\n".join(all_branch_details)
    
    prompt_summary = f"""
以下是今天各个分支的具体代码修改点列举（Markdown格式汇总）：
{combined_docs[:8000]}

请你作为技术主管，对整个项目（各分支）今天的工作进展写一份概括性技术文档风格的工作总结。
【要求格式】：
1. **工作进度概览**：一段话（不超过 50 字），高度概括各分支今天主要产出了什么成果或解决了什么痛点。
2. **各版块协同分析与点评**：基于所有分支的变动，评估当天的协同开发与代码活跃状况，或者提出一些潜在的关联提醒与影响评估。
"""
    final_summary_md = call_openclaw_llm(prompt_summary)
    if not final_summary_md:
        final_summary_md = "由于 AI 服务调用错误，本次未生成总览性点评。"

    overall_report = os.path.join(report_dir, "OVERALL_SUMMARY.md")
    with open(overall_report, "w", encoding="utf-8") as f:
        f.write(f"# 综合技术工作总结 ({date_str})\n\n{final_summary_md}\n\n## 附件：各分支改动明细（也可查阅同目录下的单独分支文件）\n{combined_docs}")
        
    print(f"Process completed successfully. Check {report_dir} directory.")

if __name__ == "__main__":
    main()
