import subprocess

import pytest

from backend import pipeline
from backend.pipeline import Clip, EncodeSettings, Filters, RenderOptions

pytestmark = pytest.mark.skipif(not pipeline.ffmpeg_available(), reason="ffmpeg not installed")


def test_probe(tiny_video):
    info = pipeline.probe(tiny_video)
    assert (info.width, info.height) == (64, 48)
    assert info.has_audio
    assert info.fps == 12
    assert info.estimated_frames == 12
    assert info.frames_in(Clip(0.5, 0.25)) == 3
    assert info.frames_in(Clip(0.75, 10)) == 3  # clamps to the end of the video


def test_opus_audio_is_transcoded_to_aac(tiny_video):
    assert pipeline._audio_args(tiny_video) == ["-c:a", "aac", "-b:a", "192k"]


def test_filters_build_ffmpeg_chains():
    assert Filters().pre() == [] and Filters().post() == []
    f = Filters(deinterlace=True, denoise=2, sharpen=True)
    assert f.pre() == ["bwdif=mode=send_frame", "hqdn3d=4:3:6:4.5"]
    assert f.post() == ["unsharp=5:5:0.6:5:5:0.0"]


def test_software_encoders_are_detected():
    assert "libx264" in pipeline.available_encoders()


def test_encode_args_quality_per_encoder_family():
    assert pipeline.encode_args(EncodeSettings(encoder="libx264", quality=20))[-2:] == ["-crf", "20"]
    nv = pipeline.encode_args(EncodeSettings(encoder="h264_nvenc", quality=20))
    assert "-cq" in nv and nv[nv.index("-cq") + 1] == "20"
    hevc = pipeline.encode_args(EncodeSettings(codec="hevc", encoder="libx265"), container="mp4")
    assert hevc[-2:] == ["-tag:v", "hvc1"]


def test_mkv_encode_args_disable_interleave_buffering():
    args = pipeline.encode_args(EncodeSettings(encoder="libx264"), container="mkv")
    assert args[-2:] == ["-max_interleave_delta", "0"]


def test_encoder_codec():
    assert pipeline.encoder_codec("libx265") == "hevc"
    assert pipeline.encoder_codec("hevc_amf") == "hevc"
    assert pipeline.encoder_codec("h264_nvenc") == "h264"


def test_encode_args_unknown_codec_fails():
    with pytest.raises(RuntimeError):
        pipeline.resolve_encoder(EncodeSettings(codec="av1"))


def test_mux_subtitles_keeps_text_subtitles(tiny_video, tmp_path):
    """The download copy is an MKV with the source subtitle streams copied."""
    srt = tmp_path / "subs.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:00,500\nHello\n", encoding="utf-8")
    source = tmp_path / "with_subs.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(tiny_video), "-i", str(srt),
         "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt", str(source)],
        check=True,
    )
    assert pipeline.has_subtitles(source)
    assert not pipeline.has_subtitles(tiny_video)

    out = tmp_path / "out.mkv"
    pipeline.mux_subtitles(tiny_video, source, out)
    assert pipeline.has_subtitles(out)
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s:0",
         "-show_entries", "stream=codec_name", "-of",
         "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert codec in ("subrip", "ass")


def test_streaming_roundtrip_with_clip_and_filters(tiny_video, tmp_path):
    """Decode a clip with a pre-filter, pass frames straight through, and
    encode with a post-filter + transcoded audio."""
    options = RenderOptions(clip=Clip(0.25, 0.5), filters=Filters(denoise=1, sharpen=True),
                            encode=EncodeSettings(encoder="libx264", quality=24))
    out = tmp_path / "out.mp4"
    reader = pipeline.open_frame_reader(tiny_video, options)
    writer = pipeline.open_frame_writer(out, 64, 48, "12", audio_source=tiny_video, options=options)
    frame_bytes = 64 * 48 * 3
    frames = 0
    while True:
        buf = reader.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        writer.stdin.write(buf)
        frames += 1
    writer.stdin.close()
    assert reader.wait() == 0 and writer.wait() == 0, writer.stderr.read()
    assert frames == 6

    info = pipeline.probe(out)
    assert info.has_audio and (info.width, info.height) == (64, 48)
    assert abs(info.duration - 0.5) < 0.15
    packets = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert packets == "6", "every decoded frame must reach the output"
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert codec == "aac"


def test_directory_roundtrip(tiny_video, tmp_path):
    frames = tmp_path / "frames"
    n = pipeline.extract_frames(tiny_video, frames, RenderOptions(clip=Clip(0, 0.5)))
    assert n == 6
    out = tmp_path / "out.mp4"
    pipeline.reassemble(frames, out, "12", tiny_video,
                        RenderOptions(clip=Clip(0, 0.5), encode=EncodeSettings(encoder="libx264")))
    info = pipeline.probe(out)
    assert info.has_audio and abs(info.duration - 0.5) < 0.15


def test_proxy_and_playability(tiny_video, tmp_path):
    assert not pipeline.is_browser_playable(tiny_video)  # it's an MKV
    proxy = tmp_path / "proxy.mp4"
    pipeline.make_proxy(tiny_video, proxy, Clip(0, 0.5))
    assert pipeline.is_browser_playable(proxy)
    assert not pipeline.probe(proxy).has_audio
