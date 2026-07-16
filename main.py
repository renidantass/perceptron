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
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
        self.bias = 0
        self.weights = []
        self.max_epochs = 100

    def train(self, data: TrainingData):
        for epoch in range(0, self.max_epochs):
            epoch_errors = 0
            for sample in data:
                predicted = self.infer(sample.inputs)

                error = sample.expected_output - predicted

                # TODO: Implementar
                self.update_weights(sample, error)
                self.update_bias(error)

                if error > 0:
                    epoch_errors += 1

            if epoch_errors <= 0:
                break

    def update_weights(self, sample: Sample, error: float):
        for i in range(0, len(sample.inputs)):
            self.weights[i] += self.learning_rate * error * sample.inputs[i]

    def update_bias(self, error: float):
        self.bias += error * self.learning_rate

    def infer(self, inputs: list[int]) -> int:
        total = 0

        for i in range(0, len(inputs)):
            total += inputs[i] * self.weights[i]

        total += self.bias

        if total >= 0:
            return 1
        
        return 0

def main():
    dataset = TrainingData()

    dataset.add(Sample([1, 1, 1], 1))
    dataset.add(Sample([0, 1, 1], 1))
    dataset.add(Sample([1, 0, 1], 1))
    dataset.add(Sample([1, 0, 0], 0))
    dataset.add(Sample([0, 1, 0], 0))
    dataset.add(Sample([0, 0, 1], 0))

    perceptron = Perceptron(learning_rate=0.02)
    perceptron.train(dataset)

    output = perceptron.infer([1,0,1])

    print(f"Saída: {output}")



if __name__ == '__main__':
    main()