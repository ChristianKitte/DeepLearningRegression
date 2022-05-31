async function run() {
    if (isArrayAndFilled(currentDataSet)) {
        let shuffled = getShuffledFeatureAndLabelColumn(currentDataSet);

        let featureTensor = tf.tensor(shuffled.features);
        let labelTensor = tf.tensor(shuffled.labels);

        featureTensor.print();
        labelTensor.print();

        let normalizedfeatures = normalizeTensor(featureTensor);
        let normalizedLabels = normalizeTensor(labelTensor);

        normalizedfeatures.print();
        normalizedLabels.print();

        let values = {
            x: shuffled.features,
            y: shuffled.labels
        }

        tfvis.render.scatterplot(
            {name: 'a'},
            {values},
            {
                xLabel: 'Features',
                yLabel: 'Labels',
                height: 300
            }
        );

    } else {
        alert("Bitte wählen Sie ein DataSet aus!");
    }
}
