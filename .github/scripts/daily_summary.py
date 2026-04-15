import os
import subprocess
import requests
import json
from datetime import datetime, timedelta

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def get_recent_commits(branch):
    # 濡傛灉鎸囧畾浜嗗叿浣撳垎鏀紙闈?all锛夛紝鍒欒幏鍙栬鍒嗘敮鏈€杩?24 灏忔椂鐨?commits锛屽弽涔嬭幏鍙栧叏閮ㄥ垎鏀殑
    if branch and branch.lower() != 'all':
        print(f"Fetching commits for branch: {branch}")
        cmd = f"git log origin/{branch} --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    else:
        print("Fetching commits for ALL branches...")
        cmd = "git log --all --since='24 hours ago' --format='%s (by %an on %D) [%h]' --name-status"
    
    latest_commits = run_command(cmd)
    return latest_commits

def call_openclaw_llm(commits_data, branch):
    # 妯℃嫙 OpenClaw / AI 妯″瀷 Base 璋冪敤鐨勫疄闄?URL锛堝瑕佺敤 OpenClaw 鎴栬€呭ぇ妯″瀷锛屾浛鎹负鐪熷疄鍙敤鐨?Endpoint锛?
    api_key = os.environ.get("OPENCLAW_API_KEY", "")
    if not api_key:
        print("璀﹀憡: 鐜鍙橀噺 OPENCLAW_API_KEY 涓虹┖銆傜郴缁熷皢鍙細杈撳嚭鍘熷 Git 鏃ュ織锛屼笉杩涜 AI 鎬荤粨銆?)
        return None

    # 姝ゅ URL 涓虹ず鑼冿紝濡傛灉鏄€氫箟鍗冮棶/鏂囧績涓€瑷€/DeepSeek 璇锋崲鍦板潃鍜屾ā鍨嬪悕
    url = "https://ai.gs88.shop/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    branch_text = f"銆恵branch}銆戝垎鏀? if branch and branch.lower() != 'all' else "銆愭墍鏈夊垎鏀€?
    prompt = f"""
璇蜂綔涓轰竴浣嶈祫娣辩殑鐮斿彂鎶€鏈富绠★紝瀵箋branch_text}杩囧幓 24 灏忔椂鍐呯殑浠ｇ爜鎻愪氦璁板綍鍜屼慨鏀瑰唴瀹硅繘琛屾繁搴︽€荤粨姹囨姤銆?

銆愬己鍒舵牸寮忓拰鍐呭瑕佹眰銆戯細
1. 鏍稿績涓氬姟姒傛嫭锛堜竴娈佃瘽锛屼笉瓒呰繃 50 瀛楋級锛氱敤鏈€绠€娲佺簿鍑嗙殑闈炴妧鏈€уぇ鐧借瘽锛屼竴鍙ヨ瘽鎬荤粨杩欐鏃堕棿涓昏瑙ｅ喅浜嗕粈涔堥棶棰橈紝鎴栬€呬笂绾夸簡浠€涔堟牳蹇冨彉鍖栵紙渚嬪锛氣€滈噸鐐逛慨澶嶄簡棣栭〉鐧藉睆宕╂簝闂锛屽苟浼樺寲浜嗗簳灞傛煡璇㈢畻娉曗€濓級銆?
2. 鍏蜂綋淇敼鐐瑰垪涓撅紙鍒嗙偣璇存槑锛夛細灏嗙悙纰庣殑浠ｇ爜鎻愪氦璁板綍鎸夌収閫昏緫鑱氬悎鎴愬姛鑳芥ā鍧楋紝骞剁畝娲佹竻鏅板湴鍒楀嚭姣忎竴涓敼鍔ㄣ€備笉瑕佸儚鏈烘缈昏瘧涓€鏍风綏鍒楁瘡涓?commit 璁板綍锛岃铻嶅悎鎴愯嚜鐒堕€昏緫銆?
3. 宸ヤ綔閲忕偣璇勶紙涓€涓ゅ彞璇濊瘎杩帮級锛氬鏁翠綋鍙樻洿鐨勪唬鐮佹椿璺冨害鎴栧伐浣滈噺杩涜鐐硅瘎銆?

浠ヤ笅鏄繃鍘?24 灏忔椂鍐呯殑鍘熷浠ｇ爜鍙樻洿鏁版嵁锛堟敞鎰忓幓绮楀彇绮撅級锛?
{commits_data[:10000]}
"""

    payload = {
        "model": "gpt-5.4", # 璇风‘淇濇偍鍦ㄤ娇鐢ㄧ殑 AI 鏈嶅姟涓婅妯″瀷鍚嶅瓨鍦?
        "messages": [
            {"role": "system", "content": "浣犳槸涓€涓珮鏁堛€佺簿缁冦€佹繁鍒绘礊瀵熸妧鏈彉鏇撮€昏緫鐨勪唬鐮佸璁′笓瀹躲€?},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"璋冪敤 AI 鍒嗘瀽鍑洪敊 details: {str(e)}")
        return None

def main():
    # 浠?Workflow 鐨勮緭鍏ヤ腑鑾峰彇鐩爣鍒嗘敮鍚嶇О锛屽鏋滄湭鎻愪緵锛堝瀹氭椂浠诲姟鏃讹級榛樿涓?all
    target_branch = os.environ.get("TARGET_BRANCH", "all").strip() or "all"
    print(f"Gathering worklogs from the last 24 hours for branch: {target_branch}...")
    
    commits = get_recent_commits(target_branch)
    
    report_dir = "branch_summaries"
    os.makedirs(report_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    branch_name_for_file = "all_branches" if target_branch.lower() == 'all' else target_branch.replace("/", "_")
    report_file = os.path.join(report_dir, f"summary_{branch_name_for_file}_{date_str}.md")
    
    branch_text = f"銆恵target_branch}銆戝垎鏀? if target_branch.lower() != 'all' else "銆愭墍鏈夊垎鏀€?

    if not commits:
        content = f"## {branch_text} 鍦ㄨ繖 24 灏忔椂鍐呮病鏈変换浣曚唬鐮佹彁浜ゃ€俓n\n澶у杈涜嫤浜嗭紒濂藉ソ浼戞伅锛?
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("鏃犳彁浜よ褰曘€?)
        return
        
    print(f"鎵惧埌浠ヤ笅鎻愪氦:\n{commits[:500]}...\n")
    
    # 璋冪敤 OpenClaw / 澶фā鍨?杩涜鏅鸿兘鍑濈粌鍜屾€荤粨
    summary = call_openclaw_llm(commits, target_branch)
    
    if summary:
        content = f"# {branch_text} 鍙樻洿鎬荤粨 ({date_str})\n\n{summary}"
    else:
        # 闄嶇骇锛氬鏋滄病鏈?AI 鏈嶅姟锛岀洿鎺ュ憟鐜拌褰?
        content = f"# {branch_text} 鍘熷淇敼璁板綍 ({date_str})\n\n```text\n{commits}\n```\n\n*(閰嶇疆鏈夋晥姝ｇ‘鐨?OPENCLAW_API_KEY 鍙?AI url 鍚庤幏鍙栨櫤鑳藉寲姒傛嫭)*"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("Report generated at:", report_file)

if __name__ == "__main__":
    main()
