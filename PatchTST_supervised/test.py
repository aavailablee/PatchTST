import subprocess
import os

# 基础配置参数
config = {
    "seq_len": 48,
    "model_name": "PatchTST",
    "root_path_name": "./dataset/",
    "data_path_name": "traffic.csv",
    "model_id_name": "ice",
    "data_name": "ice",
    "random_seed": 2021,
    "pred_len_list": [24]  # 可扩展为多组预测长度 [24, 48, 96]
}

# 固定参数配置（根据你的需求调整）
fixed_args = {
    "is_training": 1,
    "features": "MS",
    "enc_in": 4,
    "e_layers": 3,
    "n_heads": 16,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "head_dropout": 0,
    "patch_len": 16,
    "stride": 8,
    "des": "Exp",
    "train_epochs": 1,
    "patience": 10,
    "lradj": "TST",
    "pct_start": 0.2,
    "itr": 1,
    "batch_size": 24,
    "learning_rate": 0.0001
}

def run_experiment():
    # 循环执行不同预测长度
    for pred_len in config["pred_len_list"]:
        # 构造动态参数
        cmd_args = [
            "python", "-u", "run_longExp.py",
            "--random_seed", str(config["random_seed"]),
            "--is_training", str(fixed_args["is_training"]),
            "--root_path", config["root_path_name"],
            "--data_path", config["data_path_name"],
            "--model_id", f"{config['model_id_name']}_{config['seq_len']}_{pred_len}",
            "--model", config["model_name"],
            "--data", config["data_name"],
            "--features", fixed_args["features"],
            "--seq_len", str(config["seq_len"]),
            "--pred_len", str(pred_len),
            "--enc_in", str(fixed_args["enc_in"]),
            "--e_layers", str(fixed_args["e_layers"]),
            "--n_heads", str(fixed_args["n_heads"]),
            "--d_model", str(fixed_args["d_model"]),
            "--d_ff", str(fixed_args["d_ff"]),
            "--dropout", str(fixed_args["dropout"]),
            "--fc_dropout", str(fixed_args["fc_dropout"]),
            "--head_dropout", str(fixed_args["head_dropout"]),
            "--patch_len", str(fixed_args["patch_len"]),
            "--stride", str(fixed_args["stride"]),
            "--des", fixed_args["des"],
            "--train_epochs", str(fixed_args["train_epochs"]),
            "--patience", str(fixed_args["patience"]),
            "--lradj", fixed_args["lradj"],
            "--pct_start", str(fixed_args["pct_start"]),
            "--itr", str(fixed_args["itr"]),
            "--batch_size", str(fixed_args["batch_size"]),
            "--learning_rate", str(fixed_args["learning_rate"])
        ]

        # 打印可复现的命令（调试时很有用）
        print("\nExecuting command:")
        print(" ".join(cmd_args), "\n")

        # 执行命令
        process = subprocess.Popen(
            cmd_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 实时打印输出
        for line in process.stdout:
            print(line, end='')

        # 等待进程结束
        process.wait()
        if process.returncode != 0:
            print(f"\n[Error] Experiment failed for pred_len={pred_len}")
        else:
            print(f"\n[Success] Completed pred_len={pred_len}")

if __name__ == "__main__":
    # 设置工作目录到脚本所在路径（可选）
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    run_experiment()