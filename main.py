from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Sample:
    inputs: list[int]
    expected_output: int


class TrainingData:
    def __init__(self) -> None:
        self.__samples: list[Sample] = []

    def add(self, sample: Sample) -> None:
        self.__samples.append(sample)

    def __iter__(self):
        yield from self.__samples


class Perceptron:
    def __init__(self, inputs_length: int, learning_rate: float) -> None:
        self.inputs_length = inputs_length
        self.learning_rate: float = learning_rate
        self.bias: float = 0.0
        self.weights: list[float] = [0.0] * self.inputs_length
        self.max_epochs: int = 100

    def train(self, data: TrainingData) -> None:
        for _ in range(0, self.max_epochs):
            epoch_errors = 0
            for sample in data:
                predicted = self.infer(sample.inputs)

                error = sample.expected_output - predicted

                self._update_parameters(sample, error)

                if error != 0:
                    epoch_errors += 1

            if epoch_errors == 0:
                break

    def _update_parameters(self, sample: Sample, error: float) -> None:
        self._update_weights(sample, error)
        self._update_bias(error)

    def _update_weights(self, sample: Sample, error: float) -> None:
        for i in range(0, len(sample.inputs)):
            self.weights[i] += self.learning_rate * error * sample.inputs[i]

    def _update_bias(self, error: float) -> None:
        self.bias += error * self.learning_rate

    def infer(self, inputs: list[int]) -> int:
        total = 0

        if len(inputs) != self.inputs_length:
            raise ValueError("Número inválido de entradas")

        for i in range(0, len(inputs)):
            total += inputs[i] * self.weights[i]

        total += self.bias

        if total >= 0:
            return 1
        else:
            return 0

def main():
    dataset = TrainingData()

    dataset.add(Sample([1, 1, 1], 1))
    dataset.add(Sample([0, 1, 1], 1))
    dataset.add(Sample([1, 0, 1], 1))
    dataset.add(Sample([1, 0, 0], 0))
    dataset.add(Sample([0, 1, 0], 0))
    dataset.add(Sample([0, 0, 1], 0))

    perceptron = Perceptron(3, learning_rate=0.02)
    perceptron.train(dataset)

    output = perceptron.infer([1,1,1])

    print("Acertou a previsão" if output == 1 else "Errou a previsão")


if __name__ == '__main__':
    main()