# main.py
from backend import Controller_Video
from frontend import Pannello_Controllo_Video

if __name__ == '__main__':
    # 1. Istanzia la logica di business
    controller = Controller_Video()
    
    # 2. Inietta la logica nell'interfaccia utente
    app = Pannello_Controllo_Video(controller)
    
    # 3. Avvia il processo dal metodo coordinatore della GUI
    app.gui_run()