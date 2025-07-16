import gradio as gr

from run_rl_cartpole_gym import start_training, stop_training, reset_training, get_data

with gr.Blocks() as demo:
    gr.Markdown("## CartPole DQN Training Visualization")
    with gr.Row():
        start_btn = gr.Button("▶️ Start")
        stop_btn = gr.Button("⏹️ Stop")
        reset_btn = gr.Button("🔄 Reset")

    timer = gr.Timer(1)
    plot = gr.LinePlot(get_data, x="episode", y="reward", every=timer)
    label = gr.Label()

    start_btn.click(start_training, outputs=[label])
    stop_btn.click(stop_training, outputs=[label])
    reset_btn.click(reset_training, outputs=[label])

demo.launch()