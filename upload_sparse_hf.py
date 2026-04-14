import sys
from huggingface_hub import HfApi, login

def main():
    if len(sys.argv) < 2:
        print("Uso: python upload_sparse_hf.py SEU_TOKEN_AQUI")
        sys.exit(1)
        
    token = sys.argv[1]
    print("A iniciar login...")
    login(token=token)
    
    api = HfApi()
    
    print("A carregar modelo sparse para pinthoz/gus-net-bert...")
    api.upload_folder(
        folder_path="attention_app/bias/models/gus-net-bert-sparse",
        repo_id="pinthoz/gus-net-bert",
        repo_type="model",
        commit_message="Substitui modelo pelo gus-net-bert-sparse mantendo o nome do repo"
    )
    print("Sucesso! Modelo atualizado na Hugging Face.")

if __name__ == "__main__":
    main()
