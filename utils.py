import plotly.graph_objects as go

def create_gauge_chart(probability):
    
        if probability<0.3:
                color = "green"
        elif probability<0.6:
                color = "yellow"
        else:
                color = "red"
        fig = go.Figure(
                go.Indicator( mode="gauge+number",
                             value=probability *100,
                             domain={'x': [0, 1], 'y': [0, 1]},
                             title={'text': "Probability of Churn",
                                    'font': {'size': 24, "color": "black"}},
                        gauge={
                                "axis": {
                                        "range": [0, 100],
                                        "tickwidth": 1,
                                        "tickcolor": "white"
                                },
                                "bar": {"color": color},
                                "bgcolor": "rgba(0, 0, 0, 0)",
                                "borderwidth": 2,
                                "bordercolor": "white",
                                "steps": [
                                        {"range": [0, 30], "color": "green"},
                                        {"range": [30, 60], "color": "yellow"},
                                        {"range": [60, 100], "color": "red"}
                                ],
                                "threshold": {
                                        "line": {"color": "white", "width": 4},
                                        "thickness": 0.75,
                                        "value": probability *100
                                }
                        }))
        fig.update_layout(paper_bgcolor="rgba(0, 0, 0, 0)",
                                plot_bgcolor="rgba(0, 0, 0, 0)",
                                font={"color": "black"},
                                width=400, height=300,
                                margin=dict(l=20, r=20, t=50, b=20))
        
        return fig

def create_model_probability_chart(probabilities):
        models = list(probabilities.keys())
        probs= list(probabilities.values())

        fig = go.Figure(
                data=[
                        go.Bar(
                                x=probs,
                                y=models,
                                orientation='h',
                                text = [f'{p:.2%}' for p in probs],
                                textposition = 'auto'
                        )
                ]
        )
        fig.update_layout(
                title="Churn Probability by Model",
                xaxis_title="Models",
                yaxis_title="Probability",
                xaxis=dict(tickformat="%.0%", range=[0, 1]),
                height=400,
                margin= dict(l=20, r=20, t=50, b=20)
        )
        return fig
        
        



