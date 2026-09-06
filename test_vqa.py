import json
import tempfile
import unittest
from pathlib import Path

from vqa import PromptConfig, build_messages, load_few_shots, materialize_videos, parse_args
from tool_detector import FrameToolDetections, ToolDetection, ToolDetectionResult


class PromptTests(unittest.TestCase):
    def test_parse_args_reads_yaml_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "vqa.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "video: case.mp4",
                        "question: What is happening?",
                        "fps: 2.5",
                        "tool-confidence: 0.4",
                        "tool_frame_count_threshold: 2",
                        "disable_organ_classifier: true",
                    ]
                ),
                encoding="utf-8",
            )
            args = parse_args(["--config", str(config_path)])

        self.assertEqual(args.video, Path("case.mp4"))
        self.assertEqual(args.question, "What is happening?")
        self.assertEqual(args.fps, 2.5)
        self.assertIsNone(args.num_frames)
        self.assertEqual(args.tool_confidence, 0.4)
        self.assertEqual(args.tool_frame_count_threshold, 2)
        self.assertTrue(args.disable_organ_classifier)

    def test_command_line_overrides_yaml_and_exclusive_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "vqa.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "video: from_config.mp4",
                        "question: Config question",
                        "fps: 1.0",
                    ]
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--video",
                    "from_cli.mp4",
                    "--question-file",
                    "question.txt",
                    "--num-frames",
                    "12",
                ]
            )

        self.assertEqual(args.video, Path("from_cli.mp4"))
        self.assertIsNone(args.question)
        self.assertEqual(args.question_file, Path("question.txt"))
        self.assertEqual(args.num_frames, 12)
        self.assertIsNone(args.fps)

    def test_build_messages_with_context_and_few_shot(self):
        config = PromptConfig(
            system_prompt="system",
            user_prompt_template="Q: {question}\nContext: {additional_context}",
            few_shots=[{"question": "example?", "answer": "yes"}],
        )
        messages = build_messages(
            config=config,
            video_path=Path("case.mp4"),
            question="current?",
            additional_context="forceps",
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"][0]["type"], "video")
        self.assertIn("forceps", messages[-1]["content"][1]["text"])

    def test_load_few_shots_requires_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shots.json"
            path.write_text(json.dumps({"question": "not an array"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                load_few_shots(path)

    def test_materialize_videos_uses_cached_sampled_frames(self):
        import numpy as np

        path = str(Path("case.mp4").resolve())
        frames = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        metadata = {"frames_indices": [0, 9], "fps": 1.0}
        messages = [{"role": "user", "content": [{"type": "video", "path": path}]}]
        result, result_metadata = materialize_videos(
            messages,
            num_frames=2,
            fps=None,
            video_cache={path: (frames, metadata)},
        )
        self.assertIs(result[0]["content"][0]["video"], frames)
        self.assertEqual(result_metadata, [metadata])

    def test_tool_detections_are_formatted_for_every_frame(self):
        tip_detection = ToolDetection(
            class_name="needle_driver_tip",
            instrument_name="needle driver",
            component="tip",
            confidence=0.875,
            bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.4),
        )
        shaft_detection = ToolDetection(
            class_name="needle_driver_shaft",
            instrument_name="needle driver",
            component="shaft",
            confidence=0.750,
            bbox_xyxy_normalized=(0.2, 0.3, 0.4, 0.5),
        )
        forceps_detection = ToolDetection(
            class_name="bipolar_forceps_tip",
            instrument_name="bipolar forceps",
            component="tip",
            confidence=0.800,
            bbox_xyxy_normalized=(0.5, 0.6, 0.7, 0.8),
        )
        clevis_detection = ToolDetection(
            class_name="needle_driver_clevis",
            instrument_name="needle driver",
            component="clevis",
            confidence=0.925,
            bbox_xyxy_normalized=(0.2, 0.4, 0.5, 0.7),
        )
        result = ToolDetectionResult(
            frames=(
                FrameToolDetections(1, 0, 0.0, (tip_detection, shaft_detection)),
                FrameToolDetections(2, 30, 1.0, (forceps_detection, clevis_detection)),
                FrameToolDetections(3, 60, 2.0, ()),
            ),
            instrument_frame_count_threshold=1,
        )
        context = result.to_prompt_context()
        self.assertIn("needle driver (component=tip", context)
        self.assertNotIn("component=shaft", context)
        self.assertIn("bbox=[0.100, 0.200, 0.300, 0.400]", context)
        self.assertNotIn("bipolar forceps", context)
        self.assertIn("Sample 3 (source frame 60, 2.00 s): no surgical tool detected", context)
        self.assertEqual(result.detection_count, 2)
        self.assertEqual(len(result.frames[0].detections), 1)
        self.assertEqual(result.instrument_names, ("needle driver",))
        statistics = result.instrument_statistics
        self.assertEqual(statistics[0].instrument_name, "needle driver")
        self.assertEqual(statistics[0].frame_count, 2)
        self.assertAlmostEqual(statistics[0].max_confidence, 0.925)
        self.assertEqual(len(statistics), 1)

    def test_tool_frame_threshold_counts_distinct_source_frames(self):
        detection = ToolDetection(
            class_name="needle_driver_tip",
            instrument_name="needle driver",
            component="tip",
            confidence=0.9,
            bbox_xyxy_normalized=(0.1, 0.2, 0.3, 0.4),
        )
        result = ToolDetectionResult(
            frames=(
                FrameToolDetections(1, 10, 1.0, (detection,)),
                FrameToolDetections(2, 10, 1.0, (detection,)),
            ),
            instrument_frame_count_threshold=1,
        )
        self.assertEqual(result.detection_count, 0)
        self.assertEqual(result.instrument_statistics, ())

    def test_tool_instrument_names_are_empty_when_no_tools_are_detected(self):
        result = ToolDetectionResult(frames=(FrameToolDetections(1, 0, 0.0, ()),))
        self.assertEqual(result.instrument_names, ())
        self.assertEqual(result.instrument_statistics, ())


if __name__ == "__main__":
    unittest.main()
