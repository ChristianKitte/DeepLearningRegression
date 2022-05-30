/**
 * Dataset mit 5 Datensätzen als {feature, label}
 * @type {*[]}
 */
const DATA_SET_05 = [];
/**
 * Dataset mit 10 Datensätzen als {feature, label}
 * @type {*[]}
 */
const DATA_SET_10 = [];
/**
 * Dataset mit 20 Datensätzen als {feature, label}
 * @type {*[]}
 */
const DATA_SET_20 = [];
/**
 * Dataset mit 50 Datensätzen als {feature, label}
 * @type {*[]}
 */
const DATA_SET_50 = [];
/**
 * Dataset mit 100 Datensätzen als {feature, label}
 * @type {*[]}
 */
const DATA_SET_100 = [];

getDataSet("DataSet5.csv", DATA_SET_05);
getDataSet("DataSet10.csv", DATA_SET_10);
getDataSet("DataSet20.csv", DATA_SET_20);
getDataSet("DataSet50.csv", DATA_SET_50);
getDataSet("DataSet100.csv", DATA_SET_100);

/**
 * Füllt das übergebene Array mit den Werten der CSV Datei. Es wird ein mit semikolon getrenntes Array mit einem
 * Komma als Dezimalzeichen und ohne Punkt als Tausendertrennzeichen erwartet.
 * @param file Die zu verwendende CSV Datei
 * @param array Das zu füllende Array
 * @returns {Promise<void>}
 */
async function getDataSet(file, array) {
    const ds = tf.data.csv(LOCAL_DATA + file, {
            delimiter: ";"
        }
    );

    const dataArray = await ds.toArray().then(result => {
        result.forEach(function (x) {
            array.push({
                feature: parseFloat(x.x.toString().replace(",", ".")),
                label: parseFloat(x.l.toString().replace(",", ".")),
                r01: parseFloat(x.r01.toString().replace(",", ".")),
                r03: parseFloat(x.r03.toString().replace(",", "."))
            });

            if (file === "DataSet100.csv") {
                let infoString = document.getElementById("on-load-string");
                infoString.classList.replace("on-load-pending", "on-load-done");
                infoString.textContent = "Daten wurden geladen";
            }
        });

        x();
    });
}

function x() {
    let s = 1;
}
