from . import (
    plots,predictions
)


def router(page):
    if page == 'make_forecast':
        predictions.make_prediction(page)
    elif page == 'view_forecast_chart':
        plots.view_predictions(page)
        