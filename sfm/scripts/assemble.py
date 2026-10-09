"""Assemble final measurements.json (grouped, with cross-checks and layout summary)."""
import json, numpy as np
ROOT = "/home/claude/kitchen/artifacts/sfm"
raw = json.load(open(f"{ROOT}/scripts/measurements_raw.json"))
stats = json.load(open(f"{ROOT}/stats.json"))
cov = json.load(open(f"{ROOT}/coverage.json"))
V = raw["values"]


def g(k):
    v = V[k]
    return dict(value_in=v["in_"], value_mm=v["mm"], sigma_in=v["sigma_in"], sigma_mm=v["sigma_mm"],
                sigma_geom_in=v["sigma_geom_in"], sigma_scale_in=v["sigma_scale_in"], method=v["method"],
                **({"triangulated_corner_in": v["triangulated_corner_in"]} if "triangulated_corner_in" in v else {}),
                **({"per_view_in": v["per_view_in"]} if "per_view_in" in v else {}))

groups = {
 "a_countertop": ["countertop_top_height_above_floor", "left_run_countertop_thickness", "right_run_countertop_thickness",
                  "left_run_countertop_front_edge_y", "right_run_countertop_front_edge_x",
                  "countertop_front_overhang_past_left_base_face", "countertop_front_overhang_past_right_base_face"],
 "b_upper_bottoms": ["right_run_upper_bottom_height", "left_run_short_upper_over_sink_bottom_height",
                     "left_run_tall_upper_right_of_sink_bottom_height", "left_run_tall_upper_left_of_sink_bottom_height",
                     "left_run_corner_upper_bottom_height", "short_upper_bottom_minus_tall_upper_bottom"],
 "c_tops": ["left_run_crown_underside_height (top of upper boxes)", "left_run_crown_top_height",
            "right_run_crown_underside_height (top of upper boxes)", "right_run_crown_top_height",
            "pantry_cap_underside_height", "pantry_cap_top_height", "pantry_height_floor_to_cap_top"],
 "d_right_run": ["right_run_countertop_end_y (run length from W2... along W1 normal)", "right_run_drawer_stack_right_end_y",
                 "right_run_door_base_left_edge_y", "right_run_door_base_to_drawer_stack_seam_y",
                 "right_run_door_base_width", "right_run_4drawer_stack_width",
                 "right_run_upper_cab1_left_y", "right_run_upper_cab1_right_y", "right_run_upper_cab2_left_y", "right_run_upper_cab2_door_meet_y",
                 "right_run_upper_cab2_right_y", "right_run_upper_cab3_left_y", "right_run_upper_cab3_door_meet_y", "right_run_upper_cab3_right_end_y",
                 "right_run_upper_cab1_single_door_width", "right_run_upper_cab2_double_door_width", "right_run_upper_cab3_double_door_width",
                 "right_run_base_door_face_x (depth from W2 backsplash)", "right_run_upper_door_face_x (depth from W2 backsplash)",
                 "right_run_door_base_left_edge_minus_left_base_face (corner filler)"],
 "e_left_run": ["left_run_3drawer_base_right_edge_x", "left_run_sink_base_right = 3drawer_left_x", "left_run_sink_base_doors_meet_x",
                "left_run_drawer_over_door_base_right_edge = sink_base_left_x", "left_run_drawer_over_door_base_left_edge_x",
                "left_run_pantry_right_edge_x", "left_run_pantry_left_edge_x",
                "left_run_3drawer_base_width", "left_run_sink_base_width", "left_run_drawer_over_door_base_width", "left_run_pantry_width",
                "left_run_upper_corner_visible_end (meets right-run upper face)_x", "left_run_upper_tall_right_right_edge_x",
                "left_run_upper_short_right = tall_right_left_x", "left_run_upper_tall_left_right_edge = short_upper_left_x",
                "left_run_upper_tall_right_width", "left_run_upper_short_width", "left_run_upper_tall_left_width (pantry edge to short upper)",
                "left_run_base_door_face_y (depth from W1 backsplash)", "left_run_upper_door_face_y (depth from W1 backsplash)",
                "pantry_face_y (depth from W1 backsplash)", "left_run_base_door_bottom_height (toe-kick top)",
                "left_run_3drawer_right_edge_minus_right_base_face (corner filler)"],
 "f_sink": ["sink_apron_front_plane_y", "sink_apron_proud_of_base_door_face", "left_run_sink_apron_right_outer_x", "left_run_sink_apron_left_outer_x",
            "sink_apron_top_height", "sink_apron_bottom_height", "sink_apron_height_measured"],
}
out_groups = {gk: {k: g(k) for k in ks} for gk, ks in groups.items()}
missing = set(V) - {k for ks in groups.values() for k in ks}
assert not missing, missing

mods = [("left_run_upper_short_width", 30), ("right_run_upper_cab2_double_door_width", 24), ("right_run_upper_cab3_double_door_width", 24),
        ("right_run_4drawer_stack_width", 21), ("left_run_upper_tall_right_width", 18), ("left_run_upper_tall_left_width (pantry edge to short upper)", 18),
        ("left_run_3drawer_base_width", 15), ("left_run_drawer_over_door_base_width", 18), ("left_run_sink_base_width", 30), ("left_run_pantry_width", 21)]
ratios = [V[k]["in_"] / n for k, n in mods]
sc = raw["scale"]
cross = dict(
 apron_height=dict(measured_in=V["sink_apron_height_measured"]["in_"], spec_in=9.875, implied_in_per_unit=sc["s_from_apron_height"],
                   note="apron-height-implied scale differs from apron-width scale by %.2f%%" % (100 * (sc["s_from_apron_height"] / sc["in_per_unit"] - 1))),
 apron_width_edge_interpretation=dict(width_to_50pct_rounded_left_edge_in=round(sc["apron_width_alternatives_in"]["to_50pct_left_edge"], 2),
                                      width_to_gap_line_in=round(sc["apron_width_alternatives_in"]["to_gap_line"], 2),
                                      adopted="midpoint (= 30.00 in by construction); scale 1-sigma = half the spread = %.2f%%" % (100 * sc["rel_sigma"])),
 countertop_height=dict(measured_in=V["countertop_top_height_above_floor"]["in_"], standard_in=36.0,
                        implied_in_per_unit_if_36=round(sc["in_per_unit"] * 36.0 / V["countertop_top_height_above_floor"]["in_"], 4),
                        note="reported, not forced"),
 standard_cabinet_modules=dict(pairs={k: dict(measured_in=V[k]["in_"], nominal_in=n) for k, n in mods},
                               mean_ratio=round(float(np.mean(ratios)), 4), std_ratio=round(float(np.std(ratios)), 4),
                               note="independent check: nominal cabinet widths agree with the apron scale to %.2f%% on average" % (100 * (np.mean(ratios) - 1))),
 upper_box_heights=dict(tall_upper_box_in=round(V["left_run_crown_underside_height (top of upper boxes)"]["in_"] - V["left_run_tall_upper_right_of_sink_bottom_height"]["in_"], 2),
                        short_upper_box_in=round(V["left_run_crown_underside_height (top of upper boxes)"]["in_"] - V["left_run_short_upper_over_sink_bottom_height"]["in_"], 2),
                        nominal="42 / 30 (crown may overlap box tops slightly)"),
 depths=dict(left_base_face_in=V["left_run_base_door_face_y (depth from W1 backsplash)"]["in_"], right_base_face_in=V["right_run_base_door_face_x (depth from W2 backsplash)"]["in_"],
             left_upper_face_in=V["left_run_upper_door_face_y (depth from W1 backsplash)"]["in_"], right_upper_face_in=V["right_run_upper_door_face_x (depth from W2 backsplash)"]["in_"],
             nominal="24 in box + 3/4 door = 24.75 from drywall; 12 + 3/4 = 12.75 from drywall; tile face is ~0.4 in proud of drywall"),
)
layout = dict(
 note="Door/drawer-face edge positions (in). Box joints sit at the centre of the dark reveal between adjacent fronts.",
 left_run_bases_x=[["corner (blind, under right-run counter)", 0, V["right_run_base_door_face_x (depth from W2 backsplash)"]["in_"]],
                   ["filler", V["right_run_base_door_face_x (depth from W2 backsplash)"]["in_"], V["left_run_3drawer_base_right_edge_x"]["in_"]],
                   ["3-drawer base", V["left_run_3drawer_base_right_edge_x"]["in_"], V["left_run_sink_base_right = 3drawer_left_x"]["in_"]],
                   ["sink base (apron sink)", V["left_run_sink_base_right = 3drawer_left_x"]["in_"], V["left_run_drawer_over_door_base_right_edge = sink_base_left_x"]["in_"]],
                   ["drawer-over-door base", V["left_run_drawer_over_door_base_right_edge = sink_base_left_x"]["in_"], V["left_run_drawer_over_door_base_left_edge_x"]["in_"]],
                   ["tall pantry", V["left_run_pantry_right_edge_x"]["in_"], V["left_run_pantry_left_edge_x"]["in_"]]],
 left_run_uppers_x=[["corner upper (blind part hidden behind right-run uppers)", 0, V["left_run_upper_tall_right_right_edge_x"]["in_"]],
                    ["42in-tall single-door upper", V["left_run_upper_tall_right_right_edge_x"]["in_"], V["left_run_upper_short_right = tall_right_left_x"]["in_"]],
                    ["30in-tall double-door upper over sink", V["left_run_upper_short_right = tall_right_left_x"]["in_"], V["left_run_upper_tall_left_right_edge = short_upper_left_x"]["in_"]],
                    ["42in-tall single-door upper", V["left_run_upper_tall_left_right_edge = short_upper_left_x"]["in_"], V["left_run_pantry_right_edge_x"]["in_"]]],
 right_run_bases_y=[["corner (left-run base depth)", 0, V["left_run_base_door_face_y (depth from W1 backsplash)"]["in_"]],
                    ["filler", V["left_run_base_door_face_y (depth from W1 backsplash)"]["in_"], V["right_run_door_base_left_edge_y"]["in_"]],
                    ["drawer-over-door base", V["right_run_door_base_left_edge_y"]["in_"], V["right_run_door_base_to_drawer_stack_seam_y"]["in_"]],
                    ["4-drawer stack", V["right_run_door_base_to_drawer_stack_seam_y"]["in_"], V["right_run_drawer_stack_right_end_y"]["in_"]],
                    ["countertop end", V["right_run_countertop_end_y (run length from W2... along W1 normal)"]["in_"], None]],
 right_run_uppers_y=[["single-door upper", V["right_run_upper_cab1_left_y"]["in_"], V["right_run_upper_cab1_right_y"]["in_"]],
                     ["double-door upper", V["right_run_upper_cab2_left_y"]["in_"], V["right_run_upper_cab2_right_y"]["in_"]],
                     ["double-door upper (run end)", V["right_run_upper_cab3_left_y"]["in_"], V["right_run_upper_cab3_right_end_y"]["in_"]]],
)
final = dict(
 frame=raw["frame"],
 scale=dict(method="Nantucket FCFS3018W-OSD apron outer width = 30.0 in, from multi-view Manhattan line fits of the apron's left/right vertical edges",
            inches_per_model_unit=round(sc["in_per_unit"], 5), relative_sigma=round(sc["rel_sigma"], 4),
            apron_width_model_units=round(sc["apron_width_units"], 5), apron_height_model_units=round(sc["apron_height_units"], 5)),
 registration=dict(registered=stats["registered_images"], not_registered=[], mean_reprojection_error_px_fullres=round(stats["mean_reprojection_error_px_fullres"], 3),
                   num_points3D=stats["num_points3D"], mean_track_length=round(stats["mean_track_length"], 3)),
 uncertainty_note="sigma_in = hypot(sigma_geom_in, sigma_scale_in). sigma_geom: pixel noise (2 px), plane/line-fit uncertainty, wall/floor origin uncertainty. sigma_scale: 1.24% scale uncertainty (dominates for large absolute heights). Relative measurements within one image (e.g. cabinet widths) are mostly limited by sigma_geom.",
 measurements=out_groups, cross_checks=cross, layout_summary=layout, coverage_point_counts=cov["regions"])
json.dump(final, open(f"{ROOT}/measurements.json", "w"), indent=1)
print("ratios", np.round(ratios, 3), np.mean(ratios), np.std(ratios))
print(json.dumps(cross, indent=1)[:3000])
