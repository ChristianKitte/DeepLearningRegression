/**
 * Das DropDown Element zur Auswahl des DataSets
 * @type {HTMLElement}
 */
let dataSetDropDow = document.getElementById("dataset");

/**
 * Handler für das OnChange Event des Dropdown zur Anzeige der DataSets. Die Funktion wählt je nach Auswahl eines
 * der geladenen DataSets aus und füllt sechs Arrays zu Anzeige.
 *     featureArray = [];       ==> Features
 *     labelArray = [];         ==> Labels
 *     noiseArray01 = [];       ==> Noise mit Standardabweichung von 0,1
 *     noiseArray03 = [];       ==> Noise mit Standardabweichung von 0,3
 *     labelArray01 = [];       ==> Label plus Noise mit Standardabweichung von 0,1
 *     labelArray03 = [];       ==> Label plus Noise mit Standardabweichung von 0,3
 * @param event Das Event
 */
dataSetDropDow.onchange = function (event) {
    featureArray = [];
    labelArray = [];
    noiseArray01 = [];
    noiseArray03 = [];
    labelArray01 = [];
    labelArray03 = [];

    let infoString = document.getElementById("on-load-string");

    if (this.value === "1") { // 5
        DATA_SET_05.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });

        currentDataSet = DATA_SET_05;
        infoString.textContent = "DataSet 05 wurden geladen";
    } else if (this.value === "2") { // 10
        DATA_SET_10.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });

        currentDataSet = DATA_SET_10;
        infoString.textContent = "DataSet 10 wurden geladen";
    } else if (this.value === "3") { // 20
        DATA_SET_20.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });

        currentDataSet = DATA_SET_20;
        infoString.textContent = "DataSet 20 wurden geladen";
    } else if (this.value === "4") { // 50
        DATA_SET_50.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });

        currentDataSet = DATA_SET_50;
        infoString.textContent = "DataSet 50 wurden geladen";
    } else if (this.value === "5") { // 100
        DATA_SET_100.forEach(function (item) {
            featureArray.push(item.feature);
            labelArray.push(item.label);
            noiseArray01.push(item.r01);
            noiseArray03.push(item.r03);
            labelArray01.push(item.label + item.r01);
            labelArray03.push(item.label + item.r03);
        });

        currentDataSet = DATA_SET_100;
        infoString.textContent = "DataSet 100 wurden geladen";
    }

    DrawGraph("dataSetGraphFunction", "Funktionswerte der Rohdaten", featureArray, labelArray);
    DrawGraph("dataSetGraphNoise01", "Rauschen mit Standardabweichung 0,1", featureArray, noiseArray01);
    DrawGraph("dataSetGraphNoise03", "Rauschen mit Standardabweichung 0,3", featureArray, noiseArray03);
    DrawGraph("dataSetGraphFunctionNoise01", "Überlagerte Funktionswerte (0,1)", featureArray, labelArray01);
    DrawGraph("dataSetGraphFunctionNoise03", "Überlagerte Funktionswerte (0,3)", featureArray, labelArray03);
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

function getData() {
    return tf.tidy(() => {
        // Durchmischen, trennen des DataSets
        const shuffled = getShuffledFeatureAndLabelColumn(currentDataSet);
        const featureTensor = tf.tensor(shuffled.features);
        const labelTensor = tf.tensor(shuffled.labels);

        // Normalisieren von Features und Labels
        const normalisierungFeatures = normalizeTensor(featureTensor);
        const normalizedFeatures = normalisierungFeatures.normierterTensor;
        const minFeature = normalisierungFeatures.min;
        const maxFeature = normalisierungFeatures.max;

        const normalisierungLabels = normalizeTensor(labelTensor);
        const normalizedLabels = normalisierungLabels.normierterTensor;
        const minLabels = normalisierungLabels.min;
        const maxLabels = normalisierungLabels.max;

        return workDataSet = {
            featureTensor: featureTensor,
            labelTensor: labelTensor,
            //normalisierungFeatures: normalisierungFeatures,
            normalizedFeatures: normalizedFeatures,
            minFeature: minFeature,
            maxFeature: maxFeature,
            //normalisierungLabels: normalisierungLabels,
            normalizedLabels: normalizedLabels,
            minLabels: minLabels,
            maxLabels: maxLabels,
        };
    });
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
                //testModel(currentNeuralNet, currentData.normalizedFeatures, currentData.normalizedLabels);
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