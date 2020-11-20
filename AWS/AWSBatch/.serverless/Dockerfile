
    FROM justinram11/lambda:build-python3.8 AS builder
    USER root
    
    
    COPY batch-holopix-processing.zip /tmp
    RUN cd /tmp && unzip -q batch-holopix-processing.zip && rm batch-holopix-processing.zip
    
    FROM justinram11/lambda:python3.8
    COPY --from=builder /tmp /var/task/batch-holopix-processing/
    RUN rm -rf /tmp/*
    