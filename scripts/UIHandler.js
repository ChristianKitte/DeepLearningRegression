//////////////////////////////////////////////////////////////////////////////////////////
////
//// Handelt alle Button Click Ereignisse
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Gibt True zurück, wenn es sich bei dem übergebenen Parameter um ein Array mit einer
 * Länge von größer 0 handelt, es also einen Inhalt hat / nicht leer ist
 * @param arr Das zu prüfende Array
 * @returns {boolean} True, wenn es ein nicht leeres Array ist
 */
function isArrayAndFilled(arr) {
    return (Array.isArray(arr) && arr.length > 0);
}

/**
 * Handelt Click auf FFNN - erstellen und initialisiert das Erstellen. das erstellte Model wird in
 * currentNeuralNet gespeichert. Standardmäßig ist die Ein- und Augabe auf ein Unit beschränkt. Bias
 * wird immer genutzt. Zudem wird eine Beschreibungsklassen vom Typ ModelDescription erstellt und
 * in currentNeuralNetDescriptor gespeichert.
 */
function createFFNNModel() {
    const model = createModel(
        inputUnits = 1,
        outputUnits = 1,
        hiddenLayers = countHiddenLayer,
        hiddenUnits = countUnits,
        useBias = true,
        activation = getActivationSelectionString());

    currentNeuralNet = model;
    document.getElementById(DOKU_TRAIN).innerHTML = "";
    document.getElementById(DOKU_TEST).innerHTML = "";

    tfvis.show.modelSummary(document.getElementById(DOKU_MODEL), model);
}

/**
 * Handelt Click auf FFNN trainieren - In einem ersten Schritt wird das Modell nochmals erzeugt und
 * kompiliert. Anschließend wird es anhand der eingestellten Daten trainiert. Nach dem Training wird
 * das Modell getestet. Beide Vorgänge werden mit einer visuellen Ausgabe dokumentiert.
 */
async function trainAndTestModel() {
    if (isArrayAndFilled(currentRandomDataArray)) {
        // Das Model frisch erzeugen
        createFFNNModel();

        // Daten auf Basis des aktuellen DataSets zusammen stellen
        currentTrainData = getNormalizedData(currentRandomDataArray);

        // Trainieren des Models
        await trainModel(
            currentNeuralNet,
            getOptimizerInstance(),
            getLossSelectionString(),
            getMetricsByLossSelection(),
            currentTrainData.normalizedFeatures,
            currentTrainData.normalizedLabels,
            countBatch,
            countEpoch,
            DOKU_TRAIN);

        // Testen des Models
        testModel(
            currentNeuralNet,
            currentTrainData,
            DOKU_TEST);
    } else {
        alert("Bitte wählen Sie ein DataSet aus!");
    }
}

/**
 * Handelt Click auf FFNN herunterladen
 */
async function downloadFFNN() {
    await currentNeuralNet.save('downloads://my-model');
}

/**
 * Handelt Click auf FFNN hochladen
 */
async function uploadFFNN() {
    alert("Nicht implementiert...");
    //currentNeuralNet = await tf.loadLayersModel('http://model-server.domain/download/model.json');
}

/*
function loadModel() {
    const MODEL_PATH = '/models/model.json';
    const model = tf.loadLayersModel(MODEL_PATH);
    console.log("model is loaded");
    return model;
}
*/

/**
 * Stellt die Anwendung auf eines von drei vordefinierten Modellen ein
 * @param model Die Kennzahl des auszuwählenden Modells
 * (1: Underfitted,2: Bestfitted, 3: Overfitted)
 */
function setPredefinedModel(model) {
    if (model === 1) {
        // underfitted
        alert("Setze unterangepasstes Modell...");

    } else if (model === 2) {
        // bestfitted
        alert("Setze gut angepasstes Modell...");
    }
    if (model === 3) {
        // overfitted
        alert("Setze überangepasstes Modell...");
    }
}
