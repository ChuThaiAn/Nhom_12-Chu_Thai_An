import subprocess

while True:
    print("\n===== ANN DỰ ĐOÁN KHÍ THẢI CO2 =====")
    print("1. Huấn luyện mô hình")
    print("2. Dự đoán CO2")
    print("3. Đánh giá mô hình")
    print("0. Thoát")

    choice = input("Chọn: ")

    if choice == "1":
        subprocess.run(["python", "train.py"])

    elif choice == "2":
        subprocess.run(["python", "predict.py"])

    elif choice == "3":
        subprocess.run(["python", "evaluate.py"])

    elif choice == "0":
        print("Kết thúc chương trình.")
        break

    else:
        print("Lựa chọn không hợp lệ!")