/**
 * Gibt ein Plotly Scatter Plot innerhalb des übergebenen Containers aus. Hierfür werden alle auszugebenden
 * Daten als Array übergeben.
 * @param divID Die ID eines DIV Elementes als Container
 * @param title Der Titel der Ausgabe
 * @param features Die Eingabewerte (X-Werte) der Funktion
 * @param labels Die Funktionswerte (Y-Werte) der Funktion
 * @constructor
 */
function DrawGraph(divID, title, features, labels) {
    var valuesDefinition = {
        x: features,
        y: labels,
        mode: "markers",
        type: 'scatter',
        name: 'Funktionswerte'
    };

    var data = [valuesDefinition];

    var layout = {
        title: title,
        width: 500,
        height: 500,
        xaxis: {
            title: 'Eingabe (X-Wert)',
            range: [-1, 1],
            showgrid: false,
            zeroline: false
        },
        yaxis: {
            title: 'Funktionswert (Y-Wert)',
            range: [-1, 1]
        }
    };

    Plotly.newPlot(divID, data, layout);
}