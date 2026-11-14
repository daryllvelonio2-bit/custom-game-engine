//! Audio: simple playback via `rodio`.
//!
//! Keeps a single output stream and a sink. Intentionally small for low-end targets.

pub struct Audio {
    _stream: Option<rodio::OutputStream>,
    stream_handle: Option<rodio::OutputStreamHandle>,
    sink: Option<rodio::Sink>,
}

impl Default for Audio {
    fn default() -> Self {
        Self::new()
    }
}

impl Audio {
    pub fn new() -> Self {
        if let Ok((stream, stream_handle)) = rodio::OutputStream::try_default() {
            let sink = rodio::Sink::try_new(&stream_handle).ok();
            return Self {
                _stream: Some(stream),
                stream_handle: Some(stream_handle),
                sink,
            };
        }
        Self {
            _stream: None,
            stream_handle: None,
            sink: None,
        }
    }

    pub fn has_sink(&self) -> bool {
        self.sink.is_some()
    }
}
