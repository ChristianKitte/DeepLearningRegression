![image](https://user-images.githubusercontent.com/32162305/150810942-99672aac-99af-47ea-849b-ba263fae0c3f.png)

---

**Deep Learning**

**Dozent: Prof. Dr. Felix Gers (Berliner Hochschule für Technik)**

**Studiengang Medieninformatik Online MA, Sommersemester 2022**

**University of Applied Sciences Emden/Leer, Faculty of Technology, Department of Electrical Engineering and
Informatics**

---

### Einsendeaufgabe EA3 : Regression Lernen mit feed-forward Neural Network (FFNN) als Model und TensorFlow.js (TFJS) als Framework/API.

Mit einem feed-forward neural Network (FFNN) soll eine Regression der reellwertigen Funktion: y(x) = (x+0.8) * (x-0.2) * (x-0.3) * (x-0.6) im Wertebereich [-1,+1] 
durchgeführt werden. 

Zum Erzeugen der Trainingsdaten sollen N zufällige, gleich-verteilte x Werte (mit N= 5, 10, 20, 50, 100) aus dem Intervall [-1,+1] (keine Normalverteilung für die x-Werte) erzeugt und dazu y(x) berechnet werden. Zum Verrauschen wird dem Funktionswert ein normal-verteiles Rauchen (Gaussian Noise) mit einer Varianz von entweder 0.1 oder 0.3 hinzugefügt. 

[zur Webseite](https://deep-learning.ckitte.de/ea3/)

## Umsetzung

![](assets/2022-06-09-21-57-11-image.png)

![](assets/2022-06-09-21-58-20-image.png)

![](assets/2022-06-09-21-59-06-image.png)

![](assets/2022-06-09-21-59-47-image.png)

## Lösung

Jeweils das gleiche Dataset mit n= 100 DS und Nois mit Varianz von 0,3

17 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=25/Batch, 200 Epochen

![](assets/2022-06-09-20-24-49-image.png)

![](assets/2022-06-09-20-25-13-image.png)

13 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=25/Batch, 200 Epochen

![](assets/2022-06-09-20-39-09-image.png)![](assets/2022-06-09-20-38-33-image.png)

13 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=50/Batch, 400 Epochen

![](assets/2022-06-09-21-54-09-image.png)

![](assets/2022-06-09-21-54-29-image.png)

13 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=5/Batch, 200 Epochen

![](assets/2022-06-09-21-22-12-image.png)

![](assets/2022-06-09-21-21-59-image.png)

13 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=25/Batch, 404 Epochen

![](assets/2022-06-09-20-56-48-image.png)

![](assets/2022-06-09-20-56-27-image.png)

2 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=25/Batch, 200 Epochen

![](assets/2022-06-09-20-27-02-image.png)

![](assets/2022-06-09-20-26-43-image.png)

2 Layer, 10 Units, ReLu, MSE, adamax (0,01), n=25/Batch, 200 Epochen![](assets/2022-06-09-20-31-35-image.png)

![](assets/2022-06-09-20-31-13-image.png)

20 Layer, 50 Units, ReLu, MSE, adamax (0,01), n=100/Batch, 300 Epochen

![](assets/2022-06-09-20-46-10-image.png)

![](assets/2022-06-09-20-45-11-image.png)

13 Layer, 32 Units, ReLu, MSE, adamax (0,01), n=1/Batch, 404 Epochen

Sehr langsam

![](assets/2022-06-09-21-08-50-image.png)

![](assets/2022-06-09-21-08-33-image.png)

## Ergebnisse
