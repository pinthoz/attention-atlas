
    @output
    @render.ui
    def attention_flow_selector():
        res = cached_result.get()
        if not res:
            return ui.HTML("")
        tokens, *_ = res
        
        options = {"-1": "All Tokens (Full Graph)"}
        options.update({str(i): f"{tok}" for i, tok in enumerate(tokens)})
        
        # Default to -1 (All Tokens)
        selected = "-1"
            
        return ui.div(
            ui.input_select(
                "attention_flow_token",
                None,
                choices=options,
                selected=selected,
                width="100%"
            ),
            class_="focus-select",
        )
