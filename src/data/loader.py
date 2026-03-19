"""
loader.py - Đọc dữ liệu và cấu hình
"""
import pandas as pd
import yaml
import os


def load_config(yaml_path: str = "configs/params.yaml") -> dict:
    """Đọc file cấu hình YAML và trả về dictionary."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_data(path: str, config: dict = None) -> pd.DataFrame:
    """
    Đọc file CSV từ đường dẫn.

    Parameters
    ----------
    path : str
        Đường dẫn tới file CSV (có thể là đường dẫn tuyệt đối hoặc tương đối).
    config : dict, optional
        Cấu hình dự án. Nếu truyền vào, sẽ lấy đường dẫn từ config['data']['raw_path']
        khi `path` không tồn tại.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(path) and config is not None:
        # Thử lấy đường dẫn từ config
        alt_path = config.get("data", {}).get("raw_path", path)
        if os.path.exists(alt_path):
            path = alt_path

    df = pd.read_csv(path)
    print(f"[Loader] Đã tải dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    return df
