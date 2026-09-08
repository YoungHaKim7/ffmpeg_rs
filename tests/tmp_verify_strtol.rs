//! TEMPORARY verification harness for a code-review finding — delete after use.
use ffmpeg_rs::filter::FilterGraph;

#[test]
fn huge_size_3000000000() {
    let mut g = FilterGraph::new();
    let r = g.create_filter("scale", "size=3000000000x100");
    match r {
        Ok(_) => println!("PORT: scale=size=3000000000x100 ACCEPTED at init"),
        Err(e) => println!("PORT: scale=size=3000000000x100 rejected at init: {e}"),
    }
}

#[test]
fn huge_size_5000000000() {
    let mut g = FilterGraph::new();
    let r = g.create_filter("scale", "size=5000000000x100");
    match r {
        Ok(_) => println!("PORT: scale=size=5000000000x100 ACCEPTED at init"),
        Err(e) => println!("PORT: scale=size=5000000000x100 rejected at init: {e}"),
    }
}

#[test]
fn beyond_long_max_size() {
    let mut g = FilterGraph::new();
    let r = g.create_filter("scale", "size=99999999999999999999x100");
    match r {
        Ok(_) => println!("PORT: >LONG_MAX size ACCEPTED at init"),
        Err(e) => println!("PORT: >LONG_MAX size rejected at init: {e}"),
    }
}
