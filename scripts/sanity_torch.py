import torch

def main():
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        x = torch.randn(2000, 2000, device="cuda")
        y = x @ x  # operación pesada en GPU
        print("matmul_ok:", y.shape)
    else:
        x = torch.randn(2000, 2000)
        y = x @ x
        print("cpu_matmul_ok:", y.shape)

if __name__ == "__main__":
    main()