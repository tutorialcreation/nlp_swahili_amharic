import mlflow
import click
import yaml
uri = 'sqlite:///data_science.db'

def _run(entrypoint, parameters={}, params=None, source_version=None, use_cache=True):
    #existing_run = _already_ran(entrypoint, parameters, source_version)
    #if use_cache and existing_run:
    #    print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
     #   return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint,parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=params,**parameters)
    return submitted_run


@click.command()
def workflow():
    with mlflow.start_run(run_name ="audio_pipeline") as active_run:
        mlflow.set_experiment("audiopipelines")
        mlflow.set_tag("mlflow.runName", "audio_pipeline")
        
        with open(r'params.yaml') as file:
            documents = yaml.full_load(file)
        
        for item, doc in documents.items():
            if item == 'model':
                params_ = doc
        _run("audio-deep-learner",{"backend":"local","use_conda":False},
                            params=params_)
        
        
if __name__=="__main__":
    workflow()