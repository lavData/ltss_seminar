class LinearHash:
    def __init__(self):
        self.size = 2
        self.count = 0
        self.data = [[] for i in range(self.size)]
        self.threshold = 2

    def add(self, key, value):
        if self.count >= self.size * self.threshold:
            self.size *= 2
            self.rehash()
        index = hash(key) % self.size
        bucket = self.data[index]
        for i in range(len(bucket)):
            if bucket[i][0] == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.count += 1

    def get(self, key):
        index = hash(key) % self.size
        bucket = self.data[index]
        for i in range(len(bucket)):
            if bucket[i][0] == key:
                return bucket[i][1]
        return None

    def rehash(self):
        new_data = [[] for i in range(self.size)]
        for bucket in self.data:
            for key, value in bucket:
                index = hash(key) % self.size
                new_data[index].append((key, value))
        self.data = new_data

    def __str__(self):
        result = ""
        for bucket in self.data:
            for key, value in bucket:
                result += str(key) + " -> " + str(value) + "\n"
        return result
