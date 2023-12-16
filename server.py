import logging
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def determine_n_clusters(dataset: np.ndarray) -> int:
    clusters_prediction_inertia = []

    k_max = 10
    linearity_criteria = 0.9
    # TODO: adjust constants

    for k in range(1, k_max + 1):
        temp_kmeans = KMeans(n_clusters=k, n_init='auto')
        temp_kmeans.fit(dataset)

        clusters_prediction_inertia.append(temp_kmeans.inertia_)

    regression_score = []
    clusters_prediction_index = np.array(range(k_max)).reshape((-1, 1))
    clusters_prediction_inertia = np.array(clusters_prediction_inertia)

    for _ in range(k_max - 2):
        regression = LinearRegression()
        regression.fit(clusters_prediction_index, clusters_prediction_inertia)
        regression_score.append(regression.score(clusters_prediction_index, clusters_prediction_inertia))

        clusters_prediction_index = clusters_prediction_index[1:]
        clusters_prediction_inertia = clusters_prediction_inertia[1:]

    regression_score = np.array(regression_score)
    logging.debug(f'{regression_score = }')

    return np.argmax(regression_score > linearity_criteria) + 1


if __name__ == '__main__':
    x_data = [0.6363964998030558, 0.7631262949944646, 1.9214628548490054, 0.3764440261640849, 0.054172136928334336, 0.05731819870844479,
              1.456766340638167, 0.11544621959786716, 1.4467744613607951, 0.6418212614720025, 1.0493613570470124, 1.284517422843237,
              1.1194549420075972, 1.7922112363856382, 0.7709453581810977, 1.7362801066898073, 0.6195788441230601, 4.749026820521011,
              4.39951753675085, 4.869975460241041, 2.6259399525235776, 3.426579866060264, 4.464491576222658, 3.6518633980961264,
              4.200218429331725, 4.990233208202055, 3.3485998384501716, 3.7805019932912365, 4.99768138120329, 2.715301977843535,
              2.0313820721544005, 2.4672327544703028, 4.793933915470272, 4.3043931510540725, 4.104782167710695, 2.8567840182698294,
              4.754265204356853, 3.4545411296950834, 2.692563602221192, 3.8210731041101886, 1.668254570721106, 1.5686582621742398,
              1.2595508468836745, 1.9752619635588844, 1.0956580342714284, 1.1509496778963682, 1.866689065770081, 1.5275139008942644,
              1.2124473577720967, 1.066824617884955]
    y_data = [3.5575977697846906, 1.1145560243268644, 1.6001737472970112, 2.2410421263832916, 2.705298731327039, 1.9267512290575324,
              3.513032673073486, 1.1630642689773658, 1.7580745243554783, 4.04491850487407, 2.7097228831125357, 1.615669203749796,
              2.1842776127145913, 2.1679219543997763, 3.360356392875096, 3.1274875658535732, 2.5850651860310596, 2.945476722991504,
              4.135688354760968, 2.149308476589036, 3.053662470371853, 2.0628196491311996, 2.4117829759160707, 3.5893044221365997,
              2.7338228805265556, 2.1726327295745573, 2.1731472917763845, 2.992429219377858, 3.060860814679386, 2.680922940725175,
              4.391073229240715, 3.3150922198917763, 2.256448096649802, 3.770466264549294, 2.850568888358324, 0.21879368613358874,
              2.352472285665957, 0.8134604384126873, 1.323549391565341, 0.7006260415684301, 3.531071318580887, 0.6168028981311449,
              2.3581154628790326, 5.706829101427535, 0.8624581491204436, 3.4399423373416034, 1.6477085527039308, 5.6696868773099265,
              5.348772310248291, 5.709094514941997]

    X = np.array(list(zip(x_data, y_data)))

    N = determine_n_clusters(X)
    logging.info(f'Found {N} clusters')

    kmeans = KMeans(n_clusters=N, n_init='auto')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    logging.debug(f'{y_kmeans = }')

    plt.figure(num='Clustered votes')
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200)

    plt.show()
