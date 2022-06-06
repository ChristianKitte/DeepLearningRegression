/**
 * Erzeugt ein einfaches Rohmodell eines neuronalen Netzes mit einem Eingabe-, sowie einen Ausgabelayer. Der
 * Ausgabelayer kann alternativ als Flattenlayer implementiert werden. Zwischen Eingabe- und Ausgabeschicht liegen die
 * angegebene Anzahl an versteckten (Hidden) Layer. Sie verfügen über identische Vorgaben. Alle LAyer werden als Dense
 * Layer hinzugefügt.
 * @param inputUnits Die Anzahl der Eingaben und Ausgaben (wenn Flatten = False ist)
 * @param hiddenLayers Die Aazuahl der versteckten Schichten
 * @param hiddenUnits Die Anzahl der Perzeptoren je versteckter Schicht
 * @param useBias Die Angabe, ob Bias genutzt werden soll
 * @param activation Die zu verwendende Aktivierungsfunktion : (elu, hardSigmoid, linear, relu, relu6, selu,
 * sigmoid, softmax, softplus, softsign, tanh, swish, mish)
 * @param optimizerInstance Eine Instanz des zu verwendenden Optimierer an
 * @param lossFunction Die zu verwendende Verlustfunktion
 * @param metricsArray Ein Array der zu erzeugenden Metriken
 * @returns {*}
 */
function createSimpleRawModel(
    inputUnits,
    hiddenLayers,
    hiddenUnits,
    useBias,
    activation,
    optimizerInstance,
    lossFunction,
    metricsArray) {

    // Ein sequentielles, Layer basiertes Modell
    const model = tf.sequential();

    //Add Input Dense Layer
    addInputDenseLayer("Eingabelayer", model, inputUnits, hiddenUnits, useBias, activation);

    //Add Hidden Dense Layer
    for (let layer = 0; layer < hiddenLayers; layer++) {
        addDenseLayer("Hidden_Dense_Layer_" + layer.toString(), model, hiddenUnits, useBias, activation)
    }

    // Add OutputLayer
    addDenseLayer("Ausgabelayer", model, inputUnits, useBias, activation)

    model.compile({
        optimizer: optimizerInstance,
        loss: lossFunction,
        metrics: ['mse']
        //metrics: metricsArray
    });

    /*
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ["mse"],
    });*/

    return model;
}

/**
 * Fügt dem Modell eine Eingabeschicht (dense) hinzu. Der Erste Layer eines Sequentiellen Netzes muss inputShape
 * definieren. Dies Anforderung erfüllt diese Funktion.
 * https://js.tensorflow.org/api/latest/#layers.dense
 * @param name Der Name des Layer
 * @param model Das zu bearbeitende Model
 * @param inputUnits Die Anzahl der Eingabeelemente (InputShape und Units werden hiermit initialisiert)
 * @param units Die Anzahl der Units der ersten hidden Schicht
 * @param useBias True, wenn Bias verwendet werden soll
 * @param activation Die zu nutzende Aktivierungsfunktion : (elu, hardSigmoid, linear, relu, relu6, selu,
 sigmoid, softmax, softplus, softsign, tanh, swish, mish)
 */
function addInputDenseLayer(name = "Neuer Eingabelayer", model, inputUnits, units, useBias, activation) {
    model.add(
        tf.layers.dense({
            name: name,
            inputShape: [inputUnits],
            units: inputUnits,
            useBias: useBias,
            activation: activation
        }));
}

/**
 * Fügt dem Modell eine einfache versteckte Schicht (dense) hinzu
 * https://js.tensorflow.org/api/latest/#layers.dense
 * @param name Der Name des Layer
 * @param model Das zu bearbeitende Model
 * @param units Die Anzahl der Perzeptoren
 * @param useBias True, wenn Bias verwendet werden soll
 * @param activation Die zu nutzende Aktivierungsfunktion : (elu, hardSigmoid, linear, relu, relu6, selu,
 * sigmoid, softmax, softplus, softsign, tanh, swish, mish)
 */
function addDenseLayer(name = "Neuer Layer", model, units, useBias, activation) {
    model.add(
        tf.layers.dense({
            name: name,
            units: units,
            useBias: useBias,
            activation: activation
        }));
}

/**
 * Intitialisiert die Anwendung mit einem hinterlegten Underfitted Modell
 */
function createUnderfittedFfnn() {
    alert("Erzeuge underfit");
}

/**
 * Intitialisiert die Anwendung mit einem hinterlegten Overfitted Modell
 */
function createOverfittedFfnn() {
    alert("Erzeuge Overfit");
}

/**
 * Intitialisiert die Anwendung mit einem hinterlegten Bestfitted Modell
 */
function createBestfittedFfnn() {
    alert("Erzeuge bestfit");
}