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
        // Dispose all variables kept in backend engine.
        tf.disposeVariables();
        // Das Model frisch erzeugen
        createFFNNModel();

        // Daten auf Basis des aktuellen DataSets zusammen stellen
        let currentTrainData = getNormalizedData(currentRandomDataArray);

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
 * Nutzt das aktuelle Modell um mit den aktuellen Daten eine Vorhersage zu machen.
 * Hierbei wird das Modell weder neu erzeugt noch vorher trainiert.
 */
function onlyUseModel() {
    if (isArrayAndFilled(currentRandomDataArray)) {
        if (currentNeuralNet !== null) {
            // Daten auf Basis des aktuellen DataSets zusammen stellen
            let currentTrainData = getNormalizedData(currentRandomDataArray);

            document.getElementById(DOKU_TRAIN).innerHTML = "";
            document.getElementById(DOKU_TEST).innerHTML = "";

            useModel(
                currentNeuralNet,
                currentTrainData,
                DOKU_TEST);
        } else {
            alert("Kein Modell vorhanden!");
        }
    } else {
        alert("Bitte wählen Sie ein DataSet aus!");
    }
}

/**
 * Handelt Click auf FFNN herunterladen
 */
async function downloadFFNN() {
    let modelName = prompt("Mit welchen Namen soll das Modell gespeichert werden?", "mein-modell");

    if (modelName != null) {
        const x = 'downloads://' + modelName;
        await currentNeuralNet.save(x);
        alert("Das Model wurde herunter geladen...");
    }
}

/**
 * Handelt Click auf FFNN hochladen
 */
async function createFFNN() {
    const input_json = document.querySelector('#model-json'); // Modell
    const input_bin = document.querySelector('#model-bin'); // Weights

    if (input_json.files.length === 0 || input_bin.files.length === 0) {
        alert("Die Modellspezifikation oder die Gewichte sind nicht vorhanden...");
    } else {
        currentNeuralNet = await tf.loadLayersModel(
            tf.io.browserFiles([input_json.files[0], input_bin.files[0]])
        );
    }

    tfvis.show.modelSummary(document.getElementById(DOKU_MODEL), currentNeuralNet);
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
        setPredifinedValues(UNDER_FITTED);

    } else if (model === 2) {
        // bestfitted
        alert("Setze gut angepasstes Modell...");
        setPredifinedValues(BEST_FITTED);
    }
    if (model === 3) {
        // overfitted
        alert("Setze überangepasstes Modell...");
        setPredifinedValues(OVER_FITTED);
    }

    createFFNNModel();
}

/**
 * Setzt die durch das übergebene JSON definierten Werte
 * @param valueJson Das JSON mit den zu setzenden Werten (UNDER-/BEST-/OVER-FITTED)
 */
function setPredifinedValues(valueJson) {
    document.getElementById('count-layer').setAttribute('value', valueJson.Model.layers);
    countHiddenLayer = parseInt(document.getElementById('count-layer').getAttribute('value'));
    let txt1 = "Aktueller Wert: " + countHiddenLayer.toString() + " Layer";
    d3.select('#count-layer-string').text(txt1);

    document.getElementById('count-neuron').setAttribute('value', valueJson.Model.units);
    countUnits = parseInt(document.getElementById('count-neuron').getAttribute('value'));
    let txt2 = "Aktueller Wert: " + countUnits.toString() + " Units";
    d3.select('#count-neuron-string').text(txt2);

    document.getElementById('count-batch').setAttribute('value', valueJson.training.batchSize);
    countBatch = parseInt(document.getElementById('count-batch').getAttribute('value'));
    let txt4 = "Aktueller Wert: " + countBatch.toString();
    d3.select('#count-batch-string').text(txt4);

    document.getElementById('count-epoch').setAttribute('value', valueJson.training.epochs);
    countEpoch = parseInt(document.getElementById('count-epoch').getAttribute('value'));
    let txt5 = "Aktueller Wert: " + countEpoch.toString();
    d3.select('#count-epoch-string').text(txt5);

    document.getElementById("activation-type").selectedIndex = valueJson.Model.activation;
    activationSelection = document.getElementById("activation-type").selectedIndex;

    document.getElementById("optimizer-type").selectedIndex = valueJson.Model.optimizer;
    optimizerSelection = document.getElementById("optimizer-type").selectedIndex;

    document.getElementById("loss-type").selectedIndex = valueJson.Model.lost;
    lossSelection = document.getElementById("loss-type").selectedIndex;

    dataArrayDropDow.value = valueJson.training.dataset;
    dataArrayDropDow.dispatchEvent(new Event('change'));
}
