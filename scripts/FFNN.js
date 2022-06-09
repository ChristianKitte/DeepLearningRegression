//////////////////////////////////////////////////////////////////////////////////////////
////
//// Alle Funktionen und Routinen zur Erstellung und zum Betreiben eines FFNN
////
//////////////////////////////////////////////////////////////////////////////////////////

/**
 * Erzeugt ein einfaches Rohmodell eines neuronalen Netzes mit einem Eingabe-, sowie einen Ausgabelayer. Zwischen
 * Eingabe- und Ausgabeschicht liegt die angegebene Anzahl an versteckten (Hidden) Layer. Sie verfügen über
 * identische Vorgaben. Alle Layer werden als Dense Layer hinzugefügt.
 * @param inputUnits Die Anzahl der Eingaben und Ausgaben
 * @param outputUnits Die Anzahl der Eingaben und Ausgaben
 * @param hiddenLayers Die Aazuahl der versteckten Schichten
 * @param hiddenUnits Die Anzahl der Perzeptoren je versteckter Schicht
 * @param useBias Die Angabe, ob Bias genutzt werden soll
 * @param activation Die zu verwendende Aktivierungsfunktion
 * @returns {*}
 */
function createModel(
    inputUnits,
    outputUnits,
    hiddenLayers,
    hiddenUnits,
    useBias,
    activation) {

    // Ein sequentielles, Layer basiertes Modell
    const model = tf.sequential();

    //Add Input Dense Layer
    addInputDenseLayer("Eingabelayer", model, inputUnits, hiddenUnits, useBias, activation);

    //Add Hidden Dense Layer
    for (let layer = 0; layer < hiddenLayers; layer++) {
        addDenseLayer("Hidden_Dense_Layer_" + layer.toString(), model, hiddenUnits, useBias, activation)
    }

    // Add OutputLayer
    addDenseLayer("Ausgabelayer", model, outputUnits, useBias, activation)

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
            units: units,
            biasRegularizer: 'l1l2',
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
 * Kompiliert das übergebene Modell und trainiert es anschließend
 * @param model Das Model
 * @param optimizerInstance Eine Instanz des zu verwendenden Optimierer an
 * @param lossFunction Die zu verwendende Verlustfunktion
 * @param metricsArray Ein Array der zu erzeugenden Metriken
 * @param inputs Die für das Training vorgesehen Features
 * @param labels Die zugehörigen Labes
 * @param batchSize Die Größe eines einzelnen Batches je Epoche
 * @param epochs Die Anzahl der Epochen
 * @param divIdVisuelleAusgabe Die ID eine DIV Containers zur Ausgabe der Metriken. Ausgegeben werden fest 'loss'
 * und 'mse'
 */
async function trainModel(model, optimizerInstance, lossFunction, metricsArray, inputs, labels, batchSize, epochs, divIdVisuelleAusgabe) {
    model.compile({
        optimizer: optimizerInstance,
        loss: lossFunction,
        metrics: ['mse']
    });

    return await model.fit(
        inputs,
        labels, {
            batchSize: batchSize,
            epochs: epochs,
            shuffle: true,
            validationSplit: 0.3,
            callbacks: tfvis.show.fitCallbacks(
                document.getElementById(divIdVisuelleAusgabe),
                ['loss', 'mse'],
                {
                    height: 200,
                    callbacks: ['onEpochEnd']
                }
            )
        }
    );
}

/**
 * Tested das übergebene und trainierte Modell auf Basis der zugrunde liegenden Funktion als
 * GroundTruth.
 * @param model Das zu testende Modell
 * @param normalizationData Die Original und normierten Daten
 * @param divIdVisuelleAusgabe Die ID eines DIV Containers zur Ausgabe des visuellen Ergebnisses
 */
function testModel(model, normalizationData, divIdVisuelleAusgabe) {
    const {
        shuffledArray,
        normalizedFeatures,
        normalizedLabels,
        minFeature,
        maxFeature,
        minLabel,
        maxLabel
    } = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        preds.print();

        const unNormXs = xs
            .mul(maxFeature.sub(minFeature))
            .add(minFeature);

        const unNormPreds = preds
            .mul(maxLabel.sub(minLabel))
            .add(minLabel);

        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });


    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
    });

    let origX = shuffledArray.featureArray;
    let origY = shuffledArray.labelArray;
    originalPoints = [];

    origX.forEach((element, i) => {
        originalPoints.push(
            {x: element, y: origY[i]}
        );
    });

    tfvis.render.scatterplot(
        document.getElementById(divIdVisuelleAusgabe),
        {values: [originalPoints, predictedPoints], series: ['Original', 'Vorhersage']},
        {
            xLabel: 'X Wert',
            yLabel: 'Y Wert',
            height: 300
        }
    );
}
