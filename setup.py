import os

if __name__ == "__main__":
    print('KWS Transformers Env Setup')
    print('Installing required packages')
    
    command = f'pip --no-cache-dir install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 --q'
    os.system(command)
    
    command = f'pip install -r requirements.txt --q'
    os.system(command)
    
    print('Setup Complete')