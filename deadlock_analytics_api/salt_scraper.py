import json
import os

import pika

from deadlock_analytics_api.globs import CH_POOL

RMQ_HOST = os.environ.get("RMQ_HOST", "154.53.45.225")
RMQ_USER = os.environ.get("RMQ_USER", "manuel")
RMQ_PASS = os.environ.get("RMQ_PASS")


def match_salts_callback(ch, method, properties, body):
    print(f" [x] Received {body}")
    try:
        json_str = body.decode("utf-8")
        data = json.loads(json_str)
        query = """
        INSERT INTO match_salts (match_id, cluster_id, metadata_salt, replay_salt)
        VALUES (%(match_id)s, %(cluster_id)s, %(metadata_salt)s, %(replay_salt)s)
        """
        with CH_POOL.get_client() as client:
            client.execute(query, data)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [x] Done")
    except Exception as e:
        print(f" [x] Error: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=RMQ_HOST,
            credentials=pika.PlainCredentials(RMQ_USER, RMQ_PASS),
            virtual_host="public",
        )
    )
    channel = connection.channel()
    channel.basic_qos(prefetch_count=10)
    channel.basic_consume(
        queue="matchdata_salts",
        on_message_callback=match_salts_callback,
        auto_ack=False,
    )
    print(" [*] Waiting for messages. To exit press CTRL+C")
    channel.start_consuming()
