"""Run once locally to import sample_qa.json into the knowledge base."""
from pathlib import Path

from db_utils import init_db, load_json_to_kb

if __name__ == "__main__":
    init_db()
    sample_file = Path(__file__).with_name("sample_qa.json")
    count = load_json_to_kb(str(sample_file))
    print(f"成功导入 {count} 条问答，重复的问题自动跳过。")
