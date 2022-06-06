/**
 * Das DropDown Element zur Auswahl des DataSets
 * @type {HTMLElement}
 */
let dataSetDropDow = document.getElementById("dataset");

/**
 * Handelt das Event OnChange des Dropdown für die Auswahl des DataSets
 * @param event
 */
dataSetDropDow.onchange = function (event) {
    let infoString = document.getElementById("on-load-string");
    let countDataArray;

    if (this.value === "1") { // 5
        countDataArray = 5;
        infoString.textContent = "DataSet 05 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "2") { // 10
        countDataArray = 10;
        infoString.textContent = "DataSet 10 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "3") { // 20
        countDataArray = 20;
        infoString.textContent = "DataSet 20 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "4") { // 50
        countDataArray = 50;
        infoString.textContent = "DataSet 50 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    } else if (this.value === "5") { // 100
        countDataArray = 100;
        infoString.textContent = "DataSet 100 wurden geladen";
        infoString.classList.replace("on-load-pending", "on-load-done");
    }

    let dataArray = getDataArray(countDataArray);
    let dataArrayNoisy = getDataArrayWithNoise(dataArray, 0.3);

    currentDataSet = dataArrayNoisy;

    DrawGraph("dataSetGraphFunction", "Funktionswerte der Rohdaten", dataArrayNoisy);
};

// Handle Button Clicks
/**
 * Handelt Click auf FFNN erstellen und initialisiert das Erstellen. das erstellte Model wird in
 * currentNeuralNet gespeichert. Standardmäßig ist die Ein- und Augabe auf ein Unit beschränkt und es
 * wird kein Flatten Layer gesetzt. Bias wird immer genutzt. Zudem wird eine Beschreibungsklassen vom
 * Typ ModelDescription erstellt und in currentNeuralNetDescriptor gespeichert.
 */
function createAndBuildFFNN() {
    currentNeuralNet = null;

    const ffnn = createSimpleRawModel(
        inputUnits = 1,
        hiddenLayers = countHiddenLayer,
        hiddenUnits = countUnits,
        useBias = true,
        activation = getActivationSelectionString(),
        getOptimizerInstance(),
        getLossSelectionString(),
        getMetricsByLossSelection());

    printModel("dokuNetz", ffnn);
    currentNeuralNet = ffnn;

    const ffnnDescriptor = new ModelDescription(
        inputUnits = 1,
        hiddenLayers = countHiddenLayer,
        hiddenUnits = countUnits,
        useBias = true,
        activation = getActivationSelectionString(),
        getOptimizerInstance(),
        getLossSelectionString(),
        getMetricsByLossSelection());

    currentNeuralNetDescriptor = ffnnDescriptor;

    document.getElementById("dokuTrain").innerHTML = "";
    document.getElementById("train-and-test-ffnn").disabled = false;
}

/**
 * Handelt Click auf FFNN trainieren
 */
async function trainAndTestFFNN() {
    if (isArrayAndFilled(currentDataSet)) {
        createAndBuildFFNN();

        // Daten auf Basis des aktuellen DataSets zusammen stellen
        const currentData = getData();

        // Trainieren des Models
        await trainModel(currentNeuralNet,
            currentData.normalizedFeatures,
            currentData.normalizedLabels,
            countBatch,
            countEpoch,
            "dokuTrain"
        ).then(() => {
                alert("Das Training wurde beendet!");
            }
        );
    } else {
        alert("Bitte wählen Sie ein DataSet aus!");
    }
}

/**
 * Handelt Click auf FFNN herunterladen
 */
function downloadFFNN() {
    alert("Startet den Download des aktuellen FFNN");
}

/**
 * Handelt Click auf FFNN hochladen
 */
function uploadFFNN() {
    alert("Starte das Upload eines FFNN");
}

