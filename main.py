from src.data_ingestion import load_adult_dataset


def main():
    X, y = load_adult_dataset()

    print("📊 Dataset Adult cargado correctamente")
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print("\nPrimeras filas de X:")
    print(X.head())


if __name__ == "__main__":
    main()
