import random
import socket
import threading
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs

logging.basicConfig(level=logging.INFO)

viridis = colormaps['viridis']

parser = argparse.ArgumentParser(description='MediaPipe Voter Server')

parser.add_argument('port', help='Port to run server on', type=int)
parser.add_argument('-n', '--n_clusters', help='Number of clusters (if fixed)', type=int)

ANNOTATIONS = []
CLIENT_POSITIONS = {}

fig, ax = plt.subplots()


def determine_n_clusters(dataset: np.ndarray) -> int:
    clusters_prediction_inertia = []

    k_max = min(len(dataset), 10)
    linearity_criteria = 0.9

    for k in range(1, k_max + 1):
        temp_kmeans = KMeans(n_clusters=k, n_init='auto')
        temp_kmeans.fit(dataset)

        clusters_prediction_inertia.append(temp_kmeans.inertia_)

    regression_score = []
    clusters_prediction_index = np.array(range(k_max)).reshape((-1, 1))
    clusters_prediction_inertia = np.array(clusters_prediction_inertia)

    for _ in range(1, k_max - 1):
        regression = LinearRegression()
        regression.fit(clusters_prediction_index, clusters_prediction_inertia)
        regression_score.append(regression.score(clusters_prediction_index, clusters_prediction_inertia))

        clusters_prediction_index = clusters_prediction_index[1:]
        clusters_prediction_inertia = clusters_prediction_inertia[1:]

    if len(regression_score) == 0:
        return 1

    regression_score = np.array(regression_score)
    logging.debug(f'{regression_score = }')

    return np.argmax(regression_score > linearity_criteria) + 1


def cluster_dataset(dataset: np.ndarray) -> (np.ndarray, np.ndarray):
    if args.n_clusters is None:
        n = determine_n_clusters(dataset)
    else:
        n = args.n_clusters

    logging.info(f'Found {n} clusters')

    kmeans = KMeans(n_clusters=n, n_init='auto')
    kmeans.fit(dataset)
    y_kmeans = kmeans.predict(dataset)
    cluster_centers = kmeans.cluster_centers_

    logging.debug(f'{y_kmeans = }')

    return y_kmeans, cluster_centers


def process_impact(point_distribution: np.ndarray, cluster_centers: np.ndarray) -> list[int]:
    impact = []

    for i, point in enumerate(cluster_centers):
        ratio = np.sum(point_distribution == i) / len(point_distribution)
        impact.append(round(ratio * 100))

    return impact


def annotate(point_set: np.ndarray, point_impact: list[int]):
    for annotation in ANNOTATIONS:
        annotation.remove()
    ANNOTATIONS[:] = []

    for i, (point, impact) in enumerate(zip(point_set, point_impact)):
        annotation = plt.annotate(f'   {impact}%', point)
        ANNOTATIONS.append(annotation)


def simulation() -> np.ndarray:
    dataset, _ = make_blobs(n_samples=random.randint(100, 1000), centers=random.randint(2, 9), cluster_std=random.random(), random_state=0)
    return dataset


def update(frame):
    # use case 2: simulation
    # new_data = simulation()

    # use case 3: real-time data
    new_data = list(CLIENT_POSITIONS.values())

    if len(new_data) < 2:
        return

    point_distribution, cluster_centers = cluster_dataset(new_data)
    distribution_plot.set_offsets(new_data)

    distribution_plot.set_facecolors(viridis(point_distribution * 256 // len(set(point_distribution))))
    centers_plot.set_offsets(cluster_centers)

    annotate(cluster_centers, process_impact(point_distribution, cluster_centers))


def client_handler(conn: socket.socket, addr: str):
    while True:
        try:
            data = conn.recv(1024).decode('utf-8').strip()
        except ConnectionError:
            return

        logging.debug(f'{data = }')

        position = list(map(float, data.split()))
        CLIENT_POSITIONS[addr] = position


def server_init():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('0.0.0.0', args.port))
        server.listen()

        while True:
            connection, address = server.accept()
            logging.info(f'Connected by {address}')
            threading.Thread(target=client_handler, args=(connection, address)).start()


if __name__ == '__main__':
    args = parser.parse_args()

    # following use cases are mutually exclusive (feel free to comment/uncomment ones you would like to test)
    # use case 1: pre-made tricky dataset
    # x_data = [0.6363964998030558, 0.7631262949944646, 1.9214628548490054, 0.3764440261640849, 0.054172136928334336, 0.05731819870844479,
    #           1.456766340638167, 0.11544621959786716, 1.4467744613607951, 0.6418212614720025, 1.0493613570470124, 1.284517422843237,
    #           1.1194549420075972, 1.7922112363856382, 0.7709453581810977, 1.7362801066898073, 0.6195788441230601, 4.749026820521011,
    #           4.39951753675085, 4.869975460241041, 2.6259399525235776, 3.426579866060264, 4.464491576222658, 3.6518633980961264,
    #           4.200218429331725, 4.990233208202055, 3.3485998384501716, 3.7805019932912365, 4.99768138120329, 2.715301977843535,
    #           2.0313820721544005, 2.4672327544703028, 4.793933915470272, 4.3043931510540725, 4.104782167710695, 2.8567840182698294,
    #           4.754265204356853, 3.4545411296950834, 2.692563602221192, 3.8210731041101886, 1.668254570721106, 1.5686582621742398,
    #           1.2595508468836745, 1.9752619635588844, 1.0956580342714284, 1.1509496778963682, 1.866689065770081, 1.5275139008942644,
    #           1.2124473577720967, 1.066824617884955]
    # y_data = [3.5575977697846906, 1.1145560243268644, 1.6001737472970112, 2.2410421263832916, 2.705298731327039, 1.9267512290575324,
    #           3.513032673073486, 1.1630642689773658, 1.7580745243554783, 4.04491850487407, 2.7097228831125357, 1.615669203749796,
    #           2.1842776127145913, 2.1679219543997763, 3.360356392875096, 3.1274875658535732, 2.5850651860310596, 2.945476722991504,
    #           4.135688354760968, 2.149308476589036, 3.053662470371853, 2.0628196491311996, 2.4117829759160707, 3.5893044221365997,
    #           2.7338228805265556, 2.1726327295745573, 2.1731472917763845, 2.992429219377858, 3.060860814679386, 2.680922940725175,
    #           4.391073229240715, 3.3150922198917763, 2.256448096649802, 3.770466264549294, 2.850568888358324, 0.21879368613358874,
    #           2.352472285665957, 0.8134604384126873, 1.323549391565341, 0.7006260415684301, 3.531071318580887, 0.6168028981311449,
    #           2.3581154628790326, 5.706829101427535, 0.8624581491204436, 3.4399423373416034, 1.6477085527039308, 5.6696868773099265,
    #           5.348772310248291, 5.709094514941997]
    # X = np.array(list(zip(x_data, y_data)))

    # use case 2: simulation using sklearn `make_blobs`
    # X, _ = make_blobs(n_samples=100, centers=9, cluster_std=0.3, random_state=0)

    # use case 3: real-time data
    X = np.random.uniform(0, 1, (10, 2))

    distribution, centers = cluster_dataset(X)

    distribution_plot = plt.scatter(X[:, 0], X[:, 1], s=50)
    distribution_plot.set_facecolors(viridis(distribution * 256 // len(set(distribution))))
    centers_plot = plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200)

    annotate(centers, process_impact(distribution, centers))

    animation = FuncAnimation(fig, update, interval=500, cache_frame_data=False)

    fig.tight_layout()
    fig.canvas.manager.set_window_title('Real-time clustered votes')
    plt.axis('off')

    threading.Thread(target=server_init).start()

    plt.show()
