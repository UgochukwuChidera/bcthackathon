import gradio as gr
from main import demo as review_demo
from task_b import demo as rec_demo

demo = gr.TabbedInterface(
    [review_demo, rec_demo],
    ["Review Generator", "Recommendation Agent"],
    title="BCT Hackathon",
)

if __name__ == "__main__":
    demo.launch()
