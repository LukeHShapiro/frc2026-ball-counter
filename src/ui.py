"""
src/ui.py — Phase 11 + Phase 13 + Phase 14: Gradio UI

Three tabs:
  1. Main Analysis     — video upload, scoring results, flags, export (Phase 11)
  2. Driving Analysis  — per-robot style cards, metrics, velocity chart (Phase 13)
  3. Alliance Builder  — TBA picks, simulator, do-not-pick list (Phase 14)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State holder (populated by main.py before launch)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_state: dict[str, Any] = {
    "final_scores":       {},
    "score_timeline":     [],
    "driving_report":     {},
    "team_composites":    {},
    "pick_list":          [],
    "do_not_pick":        [],
    "video_path":         None,
    "annotated_video":    None,
}

STYLE_BADGE_COLOURS = {
    "DEFENSIVE":    "#FF8C00",
    "RECKLESS":     "#FF0000",
    "SMOOTH":       "#00C800",
    "DEFENCE_PROOF":"#0064FF",
}


def set_state(**kwargs: Any) -> None:
    """Update UI state. Called from main.py after each analysis phase."""
    _state.update(kwargs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1 — Main Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_main_analysis_tab() -> None:
    with gr.Tab("Main Analysis"):
        gr.Markdown("## FRC 2026 — Per-Robot Scoring Attribution")

        with gr.Row():
            video_input = gr.File(
                label="Upload Match Video (.mp4 / .mov / .avi)",
                file_types=[".mp4", ".mov", ".avi"],
            )
            run_btn = gr.Button("Run Analysis", variant="primary")

        progress_bar = gr.Textbox(
            label="Progress", value="Idle — upload a video and click Run Analysis",
            interactive=False,
        )

        gr.Markdown("### Per-Robot Score Summary")
        score_table = gr.Dataframe(
            headers=["Team", "Total", "High Conf", "Med Conf", "Low Conf"],
            label="Scores",
            interactive=False,
        )

        gr.Markdown("### Score Timeline")
        timeline_chart = gr.LinePlot(
            x="timestamp", y="score",
            title="Cumulative Score per Robot",
            tooltip=["team_number", "timestamp", "score"],
        )

        gr.Markdown("### Flagged Events (AMBIGUOUS / LOW CONFIDENCE)")
        flag_table = gr.Dataframe(
            headers=["Frame", "Team", "Method", "Confidence", "Notes"],
            label="Flags",
            interactive=False,
        )
        flag_override = gr.Dropdown(
            label="Override attribution for selected event",
            choices=[],
            interactive=True,
        )

        gr.Markdown("### Discrepancy Panel")
        discrepancy_box = gr.Textbox(
            label="Attribution vs Scoreboard",
            interactive=False,
            lines=4,
        )

        gr.Markdown("### Export")
        with gr.Row():
            export_csv_btn  = gr.Button("Export CSV")
            export_json_btn = gr.Button("Export JSON")
            export_vid_btn  = gr.Button("Export Annotated Video")

        export_status = gr.Textbox(label="Export Status", interactive=False)

        # ── Callbacks ────────────────────────────────────────────────────────

        def run_analysis(video_file):
            if video_file is None:
                return "No video uploaded.", [], None, [], ""
            return (
                "Analysis complete (run main.py for full pipeline).",
                _scores_to_table(_state["final_scores"]),
                None,
                _flags_to_table(_state["score_timeline"]),
                _discrepancy_text(),
            )

        run_btn.click(
            run_analysis,
            inputs=[video_input],
            outputs=[progress_bar, score_table, timeline_chart, flag_table, discrepancy_box],
        )

        def do_export_csv():
            from src.export import export_csv
            path = export_csv(
                _state["final_scores"],
                _state["score_timeline"],
                "data/exports/scores.csv",
            )
            return f"CSV written → {path}"

        def do_export_json():
            from src.export import export_json
            path = export_json(
                {"final_scores": _state["final_scores"],
                 "score_timeline": _state["score_timeline"]},
                "data/exports/results.json",
            )
            return f"JSON written → {path}"

        def do_export_video():
            if _state.get("annotated_video"):
                return f"Annotated video at {_state['annotated_video']}"
            return "Run analysis first."

        export_csv_btn.click(do_export_csv,  outputs=[export_status])
        export_json_btn.click(do_export_json, outputs=[export_status])
        export_vid_btn.click(do_export_video, outputs=[export_status])


def _scores_to_table(final_scores: dict) -> list[list]:
    rows = []
    for team, data in final_scores.items():
        rows.append([
            team,
            data.get("score", 0),
            data.get("high_conf", 0),
            data.get("med_conf", 0),
            data.get("low_conf", 0),
        ])
    return rows


def _flags_to_table(score_timeline: list[dict]) -> list[list]:
    rows = []
    for evt in score_timeline:
        conf = evt.get("confidence", 1.0)
        notes = evt.get("notes", "")
        if conf < 0.50 or "AMBIGUOUS" in notes.upper():
            rows.append([
                evt.get("frame_id", ""),
                evt.get("team_number", ""),
                evt.get("method", ""),
                f"{conf:.2f}",
                notes,
            ])
    return rows


def _discrepancy_text() -> str:
    # Placeholder — populated by main.py via set_state
    return "No discrepancy data yet. Run full analysis pipeline."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 2 — Driving Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_driving_analysis_tab() -> None:
    with gr.Tab("Driving Analysis"):
        gr.Markdown("## Phase 13 — Driving Style Analysis")

        refresh_btn = gr.Button("Refresh from Analysis Results")

        # ── Per-robot style cards ─────────────────────────────────────────────
        gr.Markdown("### Robot Driving Style Cards")
        style_cards = gr.HTML(value="<p>Run analysis to populate style cards.</p>")

        # ── Full metrics table ────────────────────────────────────────────────
        gr.Markdown("### Full Driving Metrics Table")
        metrics_table = gr.Dataframe(
            headers=[
                "Team", "Style", "Secondary", "Confidence",
                "Avg Vel", "Vel Var", "Max Vel", "Collisions",
                "Coll/Min", "Shadowing Evts", "Opp Half %",
                "Path Rep", "Escape Rate", "Pressure Rate",
            ],
            label="Driving Metrics",
            interactive=False,
        )

        # ── Velocity profile chart ────────────────────────────────────────────
        gr.Markdown("### Velocity Profile (Speed over Match Time)")
        velocity_chart = gr.LinePlot(
            x="frame_id", y="speed",
            title="Robot Speed per Frame",
            tooltip=["team_number", "frame_id", "speed"],
        )

        # ── Collision event timeline ──────────────────────────────────────────
        gr.Markdown("### Collision Event Timeline")
        collision_table = gr.Dataframe(
            headers=["Team", "Frame", "Opponent Track", "Pre Vel", "Post Vel"],
            label="Collisions",
            interactive=False,
        )

        # ── Shadowing event viewer ────────────────────────────────────────────
        gr.Markdown("### Shadowing Event Viewer")
        shadowing_table = gr.Dataframe(
            headers=["Team", "Start Frame", "End Frame", "Target Track", "Duration"],
            label="Shadowing Events",
            interactive=False,
        )
        jump_frame_btn = gr.Button("Jump to Selected Frame")
        jump_status    = gr.Textbox(label="Frame", interactive=False)

        # ── Export ────────────────────────────────────────────────────────────
        with gr.Row():
            export_dr_csv_btn  = gr.Button("Export Driving Report CSV")
            export_dr_json_btn = gr.Button("Export Driving Report JSON")
        export_dr_status = gr.Textbox(label="Export Status", interactive=False)

        # ── Callbacks ─────────────────────────────────────────────────────────

        def refresh_driving():
            report = _state.get("driving_report", {})
            if not report:
                return (
                    "<p>No driving data. Run analysis first.</p>",
                    [],
                    None,
                    [],
                    [],
                )
            cards_html = _build_style_cards_html(report)
            metrics_rows = _driving_metrics_to_table(report)
            return cards_html, metrics_rows, None, [], []

        refresh_btn.click(
            refresh_driving,
            outputs=[style_cards, metrics_table, velocity_chart,
                     collision_table, shadowing_table],
        )

        def do_export_dr_csv():
            from src.export import export_driving_report_csv
            path = export_driving_report_csv(
                _state.get("driving_report", {}),
                "data/exports/driving_report.csv",
            )
            return f"CSV written → {path}"

        def do_export_dr_json():
            from src.export import export_driving_report_json
            path = export_driving_report_json(
                _state.get("driving_report", {}),
                "data/exports/driving_report.json",
            )
            return f"JSON written → {path}"

        export_dr_csv_btn.click(do_export_dr_csv,  outputs=[export_dr_status])
        export_dr_json_btn.click(do_export_dr_json, outputs=[export_dr_status])

        jump_frame_btn.click(
            lambda: "Select a shadowing event row to jump to its start frame.",
            outputs=[jump_status],
        )


def _build_style_cards_html(driving_report: dict) -> str:
    parts = []
    for team, data in driving_report.items():
        style   = data.get("style", "UNKNOWN")
        second  = data.get("secondary")
        conf    = data.get("confidence", 0.0)
        scores  = data.get("style_scores", {})
        evidence = data.get("key_evidence", [])

        colour = STYLE_BADGE_COLOURS.get(style, "#888")
        badge = f'<span style="background:{colour};color:white;padding:2px 8px;border-radius:4px;font-weight:bold">{style}</span>'
        if second:
            sec_colour = STYLE_BADGE_COLOURS.get(second, "#888")
            badge += f' <span style="background:{sec_colour};color:white;padding:2px 6px;border-radius:4px;font-size:0.85em">{second}</span>'

        bars = ""
        for sname, sval in scores.items():
            sc = STYLE_BADGE_COLOURS.get(sname, "#888")
            pct = int(sval * 100)
            bars += (
                f'<div style="margin:2px 0">'
                f'<span style="width:110px;display:inline-block">{sname}</span>'
                f'<div style="display:inline-block;width:{pct}%;height:12px;background:{sc};border-radius:3px"></div>'
                f' {sval:.2f}</div>'
            )

        ev_html = "".join(f"<li>{e}</li>" for e in evidence)

        parts.append(f"""
<div style="border:1px solid #ccc;border-radius:8px;padding:12px;margin:8px 0">
  <h3>Team {team} — {badge}</h3>
  <p><b>Confidence:</b>
    <div style="display:inline-block;width:{int(conf*100)}%;height:10px;background:#4a90d9;border-radius:4px"></div>
    {conf:.2f}
  </p>
  <b>Style Scores:</b><br>{bars}
  <b>Key Evidence:</b><ul>{ev_html}</ul>
</div>""")

    return "\n".join(parts) if parts else "<p>No driving data available.</p>"


def _driving_metrics_to_table(driving_report: dict) -> list[list]:
    rows = []
    for team, data in driving_report.items():
        m = data.get("metrics", {})
        rows.append([
            team,
            data.get("style", ""),
            data.get("secondary") or "",
            f"{data.get('confidence', 0):.2f}",
            f"{m.get('avg_velocity_px_per_frame', 0):.2f}",
            f"{m.get('velocity_variance', 0):.2f}",
            f"{m.get('max_velocity', 0):.2f}",
            m.get("collision_count", 0),
            f"{m.get('collision_rate_per_minute', 0):.2f}",
            m.get("shadowing_events", 0),
            f"{m.get('time_in_opponent_half_pct', 0):.1%}",
            f"{m.get('path_repetition_score', 0):.2f}",
            f"{m.get('escape_success_rate', 0):.2f}",
            f"{m.get('scoring_under_pressure_rate', 0):.2f}",
        ])
    return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 3 — Alliance Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_alliance_builder_tab() -> None:
    with gr.Tab("Alliance Builder"):
        gr.Markdown("## Phase 14 — TBA Alliance Builder")
        gr.Markdown(
            "> **INPUT CHECKPOINT #8**: Set your TBA API key and event key "
            "in `configs/tba_config.json` before using this tab."
        )

        # ── Inputs ────────────────────────────────────────────────────────────
        with gr.Row():
            event_key_input  = gr.Textbox(label="Event Key (e.g. 2026txhou)", value="2026txhou")
            our_team_input   = gr.Textbox(label="Your Team Number", value="XXXX")
            api_status       = gr.Textbox(label="API Status", value="Not connected", interactive=False)

        strategy_sel = gr.Dropdown(
            label="Strategy",
            choices=["balanced", "score_heavy", "defensive", "safe"],
            value="balanced",
        )

        with gr.Row():
            connect_btn  = gr.Button("Connect to TBA")
            build_btn    = gr.Button("Build Pick List", variant="primary")

        # ── Pick list table ───────────────────────────────────────────────────
        gr.Markdown("### Pick List")
        pick_list_table = gr.Dataframe(
            headers=["Rank", "Team", "Composite", "Style", "OPR",
                     "Video Score/Match", "Confidence", "Warnings"],
            label="Ranked Pick List",
            interactive=False,
        )

        # ── Top 3 recommendation ──────────────────────────────────────────────
        gr.Markdown("### Top 3 Recommendation")
        recommendation_html = gr.HTML(
            value="<p>Build the pick list to see recommendations.</p>"
        )

        # ── Do Not Pick ───────────────────────────────────────────────────────
        gr.Markdown("### Do Not Pick List")
        dnp_table = gr.Dataframe(
            headers=["Team", "Reason", "Composite Score"],
            label="Do Not Pick",
            interactive=False,
        )

        # ── Team comparison tool ──────────────────────────────────────────────
        gr.Markdown("### Team Comparison")
        with gr.Row():
            team_a_input = gr.Textbox(label="Team A")
            team_b_input = gr.Textbox(label="Team B")
            compare_btn  = gr.Button("Compare")
        comparison_table = gr.Dataframe(
            headers=["Metric", "Team A", "Team B"],
            label="Side-by-side Comparison",
            interactive=False,
        )

        # ── Alliance simulator ────────────────────────────────────────────────
        gr.Markdown("### Alliance Simulator")
        with gr.Row():
            sim_t1 = gr.Textbox(label="Robot 1 Team #")
            sim_t2 = gr.Textbox(label="Robot 2 Team #")
            sim_t3 = gr.Textbox(label="Robot 3 Team #")
            sim_btn = gr.Button("Simulate")
        sim_output = gr.HTML(value="<p>Enter 3 team numbers and click Simulate.</p>")

        # ── Data coverage indicator ────────────────────────────────────────────
        gr.Markdown("### Data Coverage")
        coverage_table = gr.Dataframe(
            headers=["Team", "Has Video Data", "TBA Data Available"],
            label="Data Coverage",
            interactive=False,
        )

        # ── Export ────────────────────────────────────────────────────────────
        with gr.Row():
            export_pl_csv_btn  = gr.Button("Export Pick List CSV")
            export_pl_json_btn = gr.Button("Export Full Report JSON")
            export_dnp_btn     = gr.Button("Export Do Not Pick CSV")
        export_ab_status = gr.Textbox(label="Export Status", interactive=False)

        # ── Callbacks ─────────────────────────────────────────────────────────

        def connect_tba(event_key, our_team):
            try:
                from src.tba_client import get_event_teams
                import json as _json
                cfg_path = Path(__file__).parent.parent / "configs" / "tba_config.json"
                if not cfg_path.exists():
                    return "configs/tba_config.json not found."
                cfg = _json.loads(cfg_path.read_text())
                api_key = cfg.get("api_key", "")
                if not api_key or api_key == "YOUR_TBA_KEY_HERE":
                    return "⚠ No API key set in configs/tba_config.json"
                teams = get_event_teams(event_key)
                return f"✓ Connected — {len(teams)} teams found at {event_key}"
            except Exception as e:
                return f"✗ Connection failed: {e}"

        connect_btn.click(connect_tba, inputs=[event_key_input, our_team_input],
                          outputs=[api_status])

        def build_pick_list(event_key, our_team, strategy):
            try:
                from src.alliance_builder import (
                    build_team_composite_scores, generate_pick_list,
                    recommend_picks, recommend_do_not_pick
                )
                composites = build_team_composite_scores(
                    event_key, our_team,
                    _state.get("final_scores", {}),
                    _state.get("driving_report", {}),
                )
                _state["team_composites"] = composites

                picks = generate_pick_list(our_team, event_key, composites, top_n=20)
                _state["pick_list"] = picks

                recommendation = recommend_picks(our_team, event_key, composites, strategy)
                dnp = recommend_do_not_pick(composites)
                _state["do_not_pick"] = dnp

                pl_rows  = _pick_list_to_table(picks)
                rec_html = _recommendation_html(recommendation)
                dnp_rows = [[t["team_number"], t.get("reason",""), t.get("composite_score","")]
                            for t in dnp]
                cov_rows = [[t["team_number"],
                             "✓" if t.get("video_data") else "✗",
                             "✓"] for t in picks]

                return pl_rows, rec_html, dnp_rows, cov_rows

            except Exception as e:
                return [], f"<p>Error: {e}</p>", [], []

        build_btn.click(
            build_pick_list,
            inputs=[event_key_input, our_team_input, strategy_sel],
            outputs=[pick_list_table, recommendation_html, dnp_table, coverage_table],
        )

        def compare_teams_cb(team_a, team_b):
            from src.alliance_builder import compare_teams
            composites = _state.get("team_composites", {})
            if not composites:
                return [["No data", "—", "—"]]
            result = compare_teams(team_a, team_b, composites)
            rows = []
            for metric, vals in result.items():
                rows.append([metric, vals.get("team_a", ""), vals.get("team_b", "")])
            return rows

        compare_btn.click(compare_teams_cb, inputs=[team_a_input, team_b_input],
                          outputs=[comparison_table])

        def simulate_alliance(t1, t2, t3):
            from src.alliance_builder import simulate_alliance
            composites = _state.get("team_composites", {})
            try:
                result = simulate_alliance([t1, t2, t3], composites)
                strengths = "".join(f"<li>{s}</li>" for s in result.get("strengths", []))
                weaknesses = "".join(f"<li>{w}</li>" for w in result.get("weaknesses", []))
                return f"""
<div style='border:1px solid #ccc;padding:12px;border-radius:8px'>
  <h4>Alliance: {t1} + {t2} + {t3}</h4>
  <p><b>Projected Score:</b> {result.get('projected_score', 'N/A'):.2f}</p>
  <p><b>Synergy:</b> {result.get('style_synergy', 'N/A')}</p>
  <p><b>Overall Rating:</b> {result.get('overall_rating', 0):.2f}</p>
  <b>Strengths:</b><ul>{strengths}</ul>
  <b>Weaknesses:</b><ul>{weaknesses}</ul>
</div>"""
            except Exception as e:
                return f"<p>Simulation error: {e}</p>"

        sim_btn.click(simulate_alliance, inputs=[sim_t1, sim_t2, sim_t3],
                      outputs=[sim_output])

        def do_export_pl_csv():
            from src.export import export_pick_list_csv
            path = export_pick_list_csv(_state.get("pick_list", []),
                                        "data/exports/pick_list.csv")
            return f"CSV written → {path}"

        def do_export_pl_json():
            from src.export import export_pick_list_json
            path = export_pick_list_json(
                {"pick_list": _state.get("pick_list", []),
                 "do_not_pick": _state.get("do_not_pick", [])},
                "data/exports/alliance_report.json",
            )
            return f"JSON written → {path}"

        def do_export_dnp():
            from src.export import export_do_not_pick_csv
            path = export_do_not_pick_csv(_state.get("do_not_pick", []),
                                          "data/exports/do_not_pick.csv")
            return f"CSV written → {path}"

        export_pl_csv_btn.click(do_export_pl_csv,  outputs=[export_ab_status])
        export_pl_json_btn.click(do_export_pl_json, outputs=[export_ab_status])
        export_dnp_btn.click(do_export_dnp,         outputs=[export_ab_status])


def _pick_list_to_table(pick_list: list[dict]) -> list[list]:
    rows = []
    for entry in pick_list:
        warnings = "; ".join(entry.get("warnings", []))
        rows.append([
            entry.get("rank", ""),
            entry.get("team_number", ""),
            f"{entry.get('composite_score', 0):.3f}",
            entry.get("style", ""),
            entry.get("tba_opr", ""),
            entry.get("video_score_rate", ""),
            entry.get("data_confidence", ""),
            warnings,
        ])
    return rows


def _recommendation_html(rec: dict) -> str:
    if not rec:
        return "<p>No recommendation available.</p>"

    def _pick_card(label: str, pick: dict) -> str:
        style   = pick.get("driving_style") or "N/A"
        colour  = STYLE_BADGE_COLOURS.get(style, "#888")
        conf    = pick.get("data_confidence", "N/A")
        reasons = "".join(f"<li>{r}</li>" for r in pick.get("reasoning", []))
        return f"""
<div style='border:1px solid #ddd;padding:10px;border-radius:6px;margin:6px 0'>
  <b>{label}: Team {pick.get('team_number','?')}</b>
  <span style='background:{colour};color:white;padding:1px 6px;
               border-radius:4px;margin-left:8px'>{style}</span>
  <span style='margin-left:12px'>OPR: {pick.get('tba_opr','N/A')}</span>
  <span style='margin-left:12px'>Data confidence: {conf}</span>
  <ul>{reasons}</ul>
</div>"""

    warnings_html = "".join(
        f'<div style="background:#ffe0e0;padding:6px;border-radius:4px;margin:3px 0">⚠ {w}</div>'
        for w in rec.get("warnings", [])
    )

    return f"""
<div style='border:2px solid #4a90d9;padding:16px;border-radius:10px'>
  <h3>Recommended Alliance</h3>
  {_pick_card("Pick 1", rec.get("pick_1", {}))}
  {_pick_card("Pick 2", rec.get("pick_2", {}))}
  <p><b>Projected Alliance Score:</b> {rec.get('projected_alliance_score', 'N/A')}</p>
  <p><b>Style Synergy:</b> {rec.get('style_synergy', 'N/A')}</p>
  {warnings_html}
</div>"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# App entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_app() -> gr.Blocks:
    """
    Build and return the Gradio Blocks app with all three tabs.

    Returns:
        gr.Blocks instance ready for .launch().

    Depends on: Phase 11, Phase 13, Phase 14.
    """
    with gr.Blocks(title="FRC 2026 Ball Counter") as app:
        gr.Markdown("# FRC 2026 Ball Counter — Per-Robot Scoring Attribution")
        _build_main_analysis_tab()
        _build_driving_analysis_tab()
        _build_alliance_builder_tab()
    return app


def launch(share: bool = False, port: int = 7860) -> None:
    """
    Launch the Gradio UI.

    Args:
        share: Whether to create a public Gradio share link.
        port:  Local port to serve on.
    """
    app = build_app()
    print(f"\n[UI] Launching on http://localhost:{port}")
    app.launch(server_port=port, share=share)
