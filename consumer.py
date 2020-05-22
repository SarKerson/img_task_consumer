from kafka import KafkaConsumer

consumer = KafkaConsumer(
        'gan',
        bootstrap_servers='35.240.132.243:9092',
)

for msg in consumer:
    print(msg)
