// Handle Button Clicks

/**
 * Handelt Click auf FFNN - erstellen und initialisiert das Erstellen. das erstellte Model wird in
 * currentNeuralNet gespeichert. Standardmäßig ist die Ein- und Augabe auf ein Unit beschränkt. Bias
 * wird immer genutzt. Zudem wird eine Beschreibungsklassen vom Typ ModelDescription erstellt und
 * in currentNeuralNetDescriptor gespeichert.
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
 * Handelt Click auf FFNN trainieren - In einem ersten Schritt wird das Modell nochmals erzeugt und
 * kompiliert. Anschließend wird es anhand der eingestellten Daten trainiert. Nach dem Training wird
 * das Modell getestet. Beide Vorgänge werden mit einer visuellen Ausgabe dokumentiert.
 */
async function trainAndTestFFNN() {
    if (isArrayAndFilled(currentDataSet)) {
        createAndBuildFFNN();

        // Daten auf Basis des aktuellen DataSets zusammen stellen
        currentTrainData = getNormalizedData(currentDataSet);

        // Trainieren des Models
        await trainModel(currentNeuralNet,
            currentTrainData.normalizedFeatures,
            currentTrainData.normalizedLabels,
            countBatch,
            countEpoch,
            "dokuTrain"
        ).then(() => {
                alert("Das Training wurde beendet, fahre mit dem Testen fort!");
                testModel(currentNeuralNet, 'dokuTest');
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
    alert("Nicht implementiert...");
}

/**
 * Handelt Click auf FFNN hochladen
 */
function uploadFFNN() {
    alert("Nicht implementiert...");
}

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