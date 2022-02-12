# ChatBot2

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 9.1.6.

## Opis

Diplomska naloga, Izdelava pogovornega robota z rekurentno nevronsko mrežo LSTM.

Angularjs frontend v ozadju pa se uporabljajo modeli nevronskih mrež izdelani in naučeni v programskem jeziku Python z knjižnico Tensorflow. Modeli so za uporabo v aplikaciji pretvorjeni iz python v javascript modele.

Koda za učenje modelov se nahaja v mapi "Python"

V mapi "Koncni izdelek slike + diplomska" so dodane slike končnega izdelka in končna verzija diplomske naloge. 

## Navodila

### Lokacije pomembnih datotek

Podatkovne zbirke --> ./Datasets

Modeli --> ./src/assets/Models

### Navodila za namestitev

Potrebujete node packet manager (npm) in angular-cli.

1.) Download zip & extract ali fork repozitorija na poljubno lokacijo.

2.) CMD / PowerShell window v direktoriju projekta (za izvajanje ukazov).

3.) Ukaz 'npm i' za namestitev.

4.) Ukaz 'npm audit fix' v kolikor na koraku 3 opozori na prekrivanja različic.

5.) Ukaz 'ng serve' za zagon aplikacije na lokalnem strežniku.

6.) `http://localhost:4200/` v brskalniku.

### Development server

Run `ng serve` for a dev server. Navigate to `http://localhost:4200/`. The app will automatically reload if you change any of the source files.

### Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI README](https://github.com/angular/angular-cli/blob/master/README.md).
