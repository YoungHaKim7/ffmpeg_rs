use ffmpeg_rs::filter::parser::segment_parse;

fn main() {
    for desc in [
        "scale=320:240:0:0:qvga",
        "scale=320:240:0:0:qvga:bt709",
        "scale=320:240:0:0:qvga:-1",
        "scale=320:240:0:0:qvga:auto",
    ] {
        match segment_parse(desc) {
            Ok(seg) => {
                for chain in &seg.chains {
                    for f in &chain.filters {
                        println!(
                            "PARSE OK   {desc:32} -> {:?} opts={:?}",
                            f.filter_name, f.opts.entries
                        );
                    }
                }
            }
            Err(e) => println!("PARSE ERR  {desc:32} -> {:?}", e),
        }
    }

    // Full pipeline: parse + create + init (avfilter_graph_parse_ptr shape).
    let mut g = ffmpeg_rs::filter::FilterGraph::new();
    match g.parse_ptr("scale=320:240:0:0:qvga:bt709") {
        Ok((i, o)) => println!("E2E OK   open_inputs={} open_outputs={}", i.len(), o.len()),
        Err(e) => println!("E2E ERR  {:?}", e),
    }
}
