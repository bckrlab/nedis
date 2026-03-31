FROM mambaorg/micromamba:2.5-debian12-slim

WORKDIR /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba env create -y -n nedis -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Bake current repo into /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

# Install baked version (editable)
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NEDIS=0.0.0 \
    micromamba run -n nedis python -m pip install --no-deps -e /workspace

COPY --chown=$MAMBA_USER:$MAMBA_USER docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8888

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
