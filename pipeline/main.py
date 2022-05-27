import mlflow
import click
uri = 'sqlite:///data_science.db'

def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
    #existing_run = _already_ran(entrypoint, parameters, source_version)
    #if use_cache and existing_run:
    #    print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
     #   return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint,parameters))
    submitted_run = mlflow.run(".", entrypoint, **parameters)
    return submitted_run


@click.command()
def workflow():
    with mlflow.start_run(run_name ="ml_pipeline") as active_run:
        mlflow.set_experiment("mlpipelines")
        # mlflow.set_tracking_uri(uri)
        mlflow.set_tag("mlflow.runName", "ml_pipeline")
        _run("regression-modeling",{"backend":"local","use_conda":False})
        _run("deep-learner",{"backend":"local","use_conda":False})
        
        
if __name__=="__main__":
    workflow()